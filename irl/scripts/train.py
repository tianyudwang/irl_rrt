import argparse
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"]="1" # suppress d4rl import warning
import sys
import random
import time
import uuid
from collections import OrderedDict
from tqdm import tqdm
import h5py

import gym
import d4rl

import numpy as np
import torch

from ompl import util as ou

from irl.agents.irl_agent import IRLAgent
import irl.utils.pytorch_util as ptu
from irl.utils import utils, types, wrappers, logger

try:
    from icecream import install  # noqa
    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below

# Check if we are running python 3.8+ (supress stableBasline3 warning)
# we need to patch saved model under python 3.6/3.7 to load them
NEWER_PYTHON_VERSION = sys.version_info.major == 3 and sys.version_info.minor >= 8

CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}


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
        self.logger = logger.Logger(self.params["logdir"])

        ptu.init_gpu(use_gpu=not self.params["no_gpu"], gpu_id=self.params["which_gpu"])

        # Initialize environment and agent
        self.init_env()
        self.init_agent()

        # Set random seeds
        self.set_seeds()

        # Maximum length for episodes
        self.params["ep_len"] = self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params["ep_len"]

        # simulation timestep, will be used for video saving
        self.fps = 10
        self.save_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"../../models/{self.params['planner_type']}"
        )
        if not (os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

    def set_seeds(self):
        """Set random seeds"""
        seed = self.params["seed"]
        rng = np.random.RandomState(seed)
        env_seed = rng.randint(0, (1 << 31) - 1)
        print(f"Using random seed {env_seed} for environments")
        ou.RNG(env_seed)
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.manual_seed(env_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(env_seed)
        self.env.seed(env_seed)


    def init_env(self):
        """Initialize environments"""
        if self.params["env_name"] == "maze2d-umaze-v1":
            env = gym.make(self.params["env_name"])
            env = wrappers.Maze2DFixedStartWrapper(env)
            env = wrappers.Maze2DTransitionWrapper(env)
            env = wrappers.Maze2DFirstExitWrapper(env)
            self.env = env
            # add an env for evaluation (dense reward)
            # eval_env = gym.make("maze2d-umaze-dense-v1") 
            # eval_env = wrappers.Maze2DFixedStartWrapper(eval_env)
            # eval_env = wrappers.Maze2DTransitionWrapper(eval_env)
            # self.eval_env = eval_env
            self.eval_env = env
            
            print(f"\nInitialized environment {self.params['env_name']}")
            print(f"Eval env reference min score: {self.eval_env.unwrapped.ref_min_score}")
            print(f"Eval env reference max score: {self.eval_env.unwrapped.ref_max_score}")
            
        elif self.params["env_name"] == "antmaze-umaze-v1":
            env = gym.make(self.params["env_name"])
            env = wrappers.AntMazeFixedGoalWrapper(env)
            env = wrappers.AntMazeFixedStartWrapper(env)
            env = wrappers.AntMazeTransitionWrapper(env)
            env = wrappers.AntMazeFirstExitWrapper(env)
            self.env = env

            self.eval_env = env
        else:
            raise ValueError(f"Environment {self.params['env_name']} is not supported")


    def init_agent(self):
        """Initialize the IRL agent"""
        assert len(self.env.observation_space.shape) < 2, "Cannot handle image observations"
        ob_dim = self.env.observation_space.shape[0]

        assert isinstance(self.env.action_space, gym.spaces.Box), "Only consider continuous action space"
        ac_dim = self.env.action_space.shape[0]

        print(f"Observation space dimension {ob_dim}, action space dimension {ac_dim}\n")
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim

        self.agent = IRLAgent(
            self.env,
            self.params["agent_params"],
            self.params["planner_type"],
            self.params["timeLimit"],
        )

    def training_loop(self):

        self.start_time = time.time()

        # Collect expert demonstrations
        demo_trajectories = self.collect_demo_trajectories(
            self.params["expert_filename"], 
            self.params["demo_size"]
        )

        self.agent.add_to_buffer(demo_trajectories)

        for itr in tqdm(range(self.params["n_iter"]), dynamic_ncols=True):
            print(f"\n********** Iteration {itr} ************")

            # decide if videos should be rendered/logged at this iteration
            self.log_video = all([
                (itr + 1) % self.params["video_log_freq"] == 0,
                self.params["video_log_freq"] != -1,
            ])

            # decide if metrics should be logged
            self.logmetrics = all([
                (itr + 1) % self.params["scalar_log_freq"] == 0,
                self.params["scalar_log_freq"] != -1,
            ])

            # Train reward
            reward_logs = self.agent.train_reward()
            
            # Train actor
            policy_logs = self.agent.train_policy(self.params['timesteps_per_policy_update'])

            # log/save
            if self.log_video or self.logmetrics:
                model_dir = os.path.join(self.params['logdir'], f"models_itr_{itr:02d}")
                if not (os.path.exists(model_dir)):
                    os.makedirs(model_dir, exist_ok=True)
                self.agent.reward.save(os.path.join(model_dir, "reward.pt"))
                self.agent.actor.save(os.path.join(model_dir, "SAC"))
                # perform logging
                print("\nBeginning logging procedure...")
                self.perform_logging(
                    itr,
                    self.agent.actor,
                    reward_logs,
                    policy_logs,
                )
                if self.params["save_params"]:
                    self.agent.save(f"{self.params['logdir']}/agent_itr_{itr}.pt")

    def collect_demo_trajectories(self, expert_filename: str, batch_size: int):
        """Load demonstration trajectories"""
        dataset = h5py.File(expert_filename, 'r')
        assert dataset['observations'].shape[1] == self.env.observation_space.shape[0], (
            f"Demonstration observation dimension {dataset['observations'].shape[1]} "
            "does not match environment observation space dimension " 
            f"{self.env.observation_space.shape[0]}"
        )
        assert dataset['actions'].shape[1] == self.env.action_space.shape[0], (
            f"Demonstration action dimension {dataset['actions'].shape[1]} "
            "does not match environment action space dimension "
            f"{self.env.action_space.shape[0]}"
        )

        utils.check_valid(dataset)

        dones = np.where(dataset['terminals'])[0][:batch_size]

        trajectories = []
        start = 0
        for end in dones:
            states = dataset['observations'][start:end+1]
            actions = dataset['actions'][start:end]
            trajectory = types.Trajectory(states, actions)
            trajectories.append(trajectory)
            start = end + 1
            # utils.render_trajectory(self.env, states[:, :2], states[:, 2:])

        print(f"Loaded {batch_size} trajectories of "
              f"{np.sum([len(traj) for traj in trajectories])} transitions")
        return trajectories


    def perform_logging(
        self, itr, eval_policy, reward_logs, policy_logs
    ):

        last_log = reward_logs[-1]

        #######################

        # Evaluate the agent policy in true environment
        # TODO: (Yifan) I add an eval env with dense reward.
        print("\nCollecting data for eval...")
        eval_paths = utils.sample_trajectories(
            self.eval_env, eval_policy, self.params["eval_batch_size"], render=False
        )

        # save eval rollouts as videos in tensorboard event file
        if self.log_video:
            eval_video_paths = utils.sample_trajectories(
                self.env, eval_policy, MAX_NVIDEO, render=True
            )

            # save train/eval videos
            print("\nSaving eval rollouts as videos...")
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
            eval_returns = [np.sum(eval_path.rewards) for eval_path in eval_paths]

            # episode lengths, for logging
            eval_ep_lens = [len(eval_path) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

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
    parser.add_argument("--env_name", type=str, default="maze2d-umaze-v1")
    parser.add_argument("--expert_filename", type=str, default="maze2d-umaze-v1.hdf5")
    parser.add_argument("--timeLimit", type=float, default=2)
    
    # parser.add_argument(
    #     "--expert_policy",
    #     type=str,
    #     choices=[
    #         "SAC_NavEnv-v0",
    #         "SAC_Pendulum-v0",
    #         "SAC_PointUMaze-v0",
    #         "TQC_Pendulum-v0",
    #         "TQC_PointUMaze-v0",
    #     ],
    # )
    parser.add_argument(
        "--planner_type",
        "-pt",
        type=str,
        choices=["rrt", "sst", "rrtstar", "prmstar"],
        default="rrtstar"
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
        "--timesteps_per_policy_update",
        type=int,
        default=15000,
        help="Number of policy updates per iteration",
    )
    parser.add_argument(
        "--transitions_per_reward_update",
        type=int,
        default=128,
        help="Number of agent transitions per reward update",
    )
    parser.add_argument(
        "--agent_actions_per_demo_transition",
        type=int,
        default=1,
        help="Number of agent actions sampled for each expert_transition",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Number of policy rollouts for evaluation",
    )

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=64)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=12)
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
        + args.planner_type
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
        + "_"
        + f"{uuid.uuid4().hex[:3]}"
    )
    logdir = os.path.join(data_path, logdir)
    params["logdir"] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir, exist_ok=True)

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
