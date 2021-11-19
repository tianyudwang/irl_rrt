import argparse
import os
import sys
import random
import time
import uuid
from collections import OrderedDict

import gym
import numpy as np
import torch

from tqdm import tqdm
from ompl import util as ou


from irl.agents.irl_agent import IRL_Agent
import irl.scripts.pytorch_util as ptu
import irl.scripts.utils as utils
from irl.scripts.logger import Logger

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below

# Check if we are running python 3.8+ (supress stableBasline3 warning)
# we need to patch saved model under python 3.6/3.7 to load them
NEWER_PYTHON_VERSION = sys.version_info.major == 3 and sys.version_info.minor >= 8

CUSTOM_OBJECTS = (
    {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    if NEWER_PYTHON_VERSION
    else {}
)


class Trainer:
    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            "n_layers": params["n_layers"],
            "size": params["size"],
            "output_size": params["output_size"],
            "learning_rate": params["learning_rate"],
        }

        train_args = {
            "reward_updates_per_iter": params["reward_updates_per_iter"],
            "transitions_per_reward_update": params["transitions_per_reward_update"],
            "agent_actions_per_demo_transition": params[
                "agent_actions_per_demo_transition"
            ],
        }

        agent_params = {**computation_graph_args, **train_args}

        self.params = params
        self.params["agent_params"] = agent_params

        #############
        ## INIT
        #############
        # Get params, create logger
        self.logger = Logger(self.params["logdir"])

        # Set random seeds
        seed = self.params["seed"]
        ou.RNG(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        ptu.init_gpu(use_gpu=not self.params["no_gpu"], gpu_id=self.params["which_gpu"])

        # Initialize environment and agent
        self.init_env()
        self.init_agent()

        # Maximum length for episodes
        self.params["ep_len"] = self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params["ep_len"]

        # simulation timestep, will be used for video saving
        self.fps = 10
        self.save_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../models"
        )

    def init_env(self):
        """
        Initialize the orgianl environment.
        """
        if self.params["env_name"] == "NavEnv-v0":
            import gym_nav

            self.env = gym.make(self.params["env_name"])
        elif self.params["env_name"] == "Pendulum-v0":
            from pendulum_env_wrapper import PendulumWrapper

            env = gym.make(self.params["env_name"])
            # *Note we only modify the Pendulum-v0 to swap the x and y axis.
            self.env = PendulumWrapper(env)

        elif self.params["env_name"] in ["PointUMaze-v0", "PointUMaze-v1"]:
            # v0 adn v1 has different reward function.
            # since we are not using the true rewad, they are the same.
            import mujoco_maze
            from remove_timeDim_wrapper import RemovTimeFeatureWrapper

            # * This env includes the time at the last axis, which should be removed.
            self.env = RemovTimeFeatureWrapper(gym.make(self.params["env_name"]))
            # self.env = gym.make(self.params["env_name"])

        else:
            raise ValueError(f"Environment {self.params['env_name']} is not supported")

    def init_agent(self):
        """
        Initialize the IRL agent.
        """
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2
        # Observation and action sizes
        ob_dim = (
            self.env.observation_space.shape
            if img
            else self.env.observation_space.shape[0]
        )
        # * We only consider envriment with continuous action space
        ac_dim = self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim

        self.agent = IRL_Agent(
            self.env,
            self.params["agent_params"],
            self.params["env_name"],
            self.params["planner_type"],
        )

    def training_loop(self):

        self.start_time = time.time()

        # Collect expert demonstrations
        demo_paths = self.collect_demo_trajectories(
            self.params["expert_policy"], self.params["demo_size"]
        )
        self.agent.add_to_buffer(demo_paths, demo=True)

        for itr in tqdm(range(self.params["n_iter"]), dynamic_ncols=True):
            print(f"\n********** Iteration {itr} ************")

            # decide if videos should be rendered/logged at this iteration
            self.log_video = all(
                [
                    (itr + 1) % self.params["video_log_freq"] == 0,
                    self.params["video_log_freq"] != -1,
                ]
            )

            # decide if metrics should be logged
            self.logmetrics = all(
                [
                    (itr + 1) % self.params["scalar_log_freq"] == 0,
                    self.params["scalar_log_freq"] != -1,
                ]
            )

            # Collect agent demonstrations
            agent_paths, train_video_paths = self.collect_agent_trajectories(
                self.agent.actor, self.params["demo_size"]
            )

            self.agent.add_to_buffer(agent_paths)

            # Train Reward
            reward_logs = self.agent.train_reward()
            for step in range(self.params["policy_updates_per_iter"]):
                policy_logs = self.agent.train_policy()

            # log/save
            if self.log_video or self.logmetrics:
                self.agent.actor.save(
                    os.path.join(
                        self.save_dir, f"SAC_{self.params['env_name']}_itr_{itr}"
                    )
                )
                # perform logging
                print("\nBeginning logging procedure...")
                self.perform_logging(
                    itr,
                    agent_paths,
                    self.agent.actor,
                    train_video_paths,
                    reward_logs,
                    policy_logs,
                )
                if self.params["save_params"]:
                    self.agent.save(f"{self.params['logdir']}/agent_itr_{itr}.pt")

    def collect_demo_trajectories(self, expert_policy, batch_size: int):
        """
        :param expert_policy:  relative path to saved expert policy
        :return:
            paths: a list of trajectories
        """

        assert isinstance(batch_size, int) and batch_size > 0
        expert_algo = expert_policy[:3].lower()

        if expert_algo == "sac":
            from stable_baselines3 import SAC

            expert_algo = SAC
        elif expert_algo == "tqc":
            from sb3_contrib import TQC

            expert_algo = TQC
        elif expert_algo == "ppo":
            from stable_baselines3 import PPO

            expert_algo = PPO
        else:
            raise ValueError(f"Expert algorithm {expert_algo} is not supported.")

        expert_policy = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../rl-trained-agents",
            expert_policy,
        )
        expert_policy = expert_algo.load(
            expert_policy, device=ptu.device, custom_objects=CUSTOM_OBJECTS
        )
        print("\nRunning expert policy to collect demonstrations...")
        demo_paths = utils.sample_trajectories(self.env, expert_policy, batch_size)
        # demo_paths = utils.pad_absorbing_states(demo_paths)
        return demo_paths

    def collect_agent_trajectories(self, collect_policy, batch_size):
        """
        :param collect_policy:  the current policy which we use to collect data
        :param batch_size:  the number of trajectories to collect
        :return:
            paths: a list trajectories
            train_video_paths: paths which also contain videos for visualization purposes

        """
        print("\nCollecting agent trajectories to be used for training...")
        paths = utils.sample_trajectories(self.env, collect_policy, batch_size)
        # paths = utils.pad_absorbing_states(paths)

        train_video_paths = None
        if self.log_video:
            print("\nCollecting train rollouts to be used for saving videos...")
            train_video_paths = utils.sample_trajectories(
                self.env, collect_policy, MAX_NVIDEO, render=True
            )
        return paths, train_video_paths

    def perform_logging(
        self, itr, paths, eval_policy, train_video_paths, reward_logs, policy_logs
    ):

        last_log = reward_logs[-1]

        #######################

        # Evaluate the agent policy in true environment
        print("\nCollecting data for eval...")
        eval_paths = utils.sample_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], render=False
        )

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            eval_video_paths = utils.sample_trajectories(
                self.env, eval_policy, MAX_NVIDEO, render=True
            )

            # save train/eval videos
            print("\nSaving train and eval rollouts as videos...")
            self.logger.log_paths_as_videos(
                train_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="train_rollouts",
            )
            self.logger.log_paths_as_videos(
                eval_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

        #######################

        # save eval metrics
        # TODO: should parse the reward training loss and policy training loss
        # TODO: should add a visualization tool to check the trained reward function
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="NavEnv-v0")
    parser.add_argument("--exp_name", type=str, default="NavEnv-v0")
    parser.add_argument(
        "--expert_policy",
        type=str,
        choices=[
            "SAC_NavEnv-v0",
            "SAC_Pendulum-v0",
            "SAC_PointUMaze-v0",
            "TQC_Pendulum-v0",
            "TQC_PointUMaze-v0",
        ],
        default="SAC_NavEnv-v0",
    )
    parser.add_argument(
        "--planner_type",
        "-pt",
        type=str,
        choices=["rrt", "sst", "rrt*", "rrtstar", "prm*", "prmstar"],
        required=True,
    )
    parser.add_argument(
        "--n_iter", "-n", type=int, default=100, help="Number of total iterations"
    )
    parser.add_argument(
        "--demo_size",
        type=int,
        default=10,
        help="Number of expert paths to add to replay buffer",
    )
    parser.add_argument(
        "--reward_updates_per_iter",
        type=int,
        default=10,
        help="Number of reward updates per iteration",
    )
    parser.add_argument(
        "--policy_updates_per_iter",
        type=int,
        default=10,
        help="Number of policy updates per iteration",
    )
    parser.add_argument(
        "--transitions_per_reward_update",
        type=int,
        default=100,
        help="Number of agent transitions per reward update",
    )
    parser.add_argument(
        "--agent_actions_per_demo_transition",
        type=int,
        default=1,
        help="Number of agent actions sampled for each expert_transition",
    )
    #    parser.add_argument(
    #        '--rrt_runs', type=int, default=1,
    #        help='Number of RRT* runs to estimate cost to go'
    #    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=10,
        help="Number of policy rollouts for evaluation",
    )

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=64)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--save_params", action="store_true")

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")
    )

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
        + "_"
        + f"{uuid.uuid4().hex[:3]}"
    )
    logdir = os.path.join(data_path, logdir)
    params["logdir"] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    #####################
    ### RUN TRAINING  ###
    #####################

    try:
        trainer = Trainer(params)
        trainer.training_loop()
    except KeyboardInterrupt:
        keep = input("\nExiting from training early.\nKeep logs & models? ([y]/n)? ")
        if keep.lower() in ["n", "no"]:
            import shutil

            shutil.rmtree(logdir)


if __name__ == "__main__":
    main()
