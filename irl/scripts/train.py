from typing import List
import argparse
import os
import time
from collections import OrderedDict

import gym
import numpy as np 
import torch 

from irl.agents.irl_agent import IRL_Agent
import irl.utils.pytorch_util as ptu 
import irl.utils.utils as utils
from irl.utils.logger import Logger
from irl.utils.wrappers import ReacherWrapper


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class Trainer():

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'output_size': params['output_size'],
            'learning_rate': params['learning_rate'],
        }

        train_args = {
            'reward_updates_per_itr': params['reward_updates_per_itr'],
            # 'transitions_per_reward_update': params['transitions_per_reward_update'],
            'transitions_per_itr': params['transitions_per_itr'],
            'agent_actions_per_demo_transition': params['agent_actions_per_demo_transition']
        }

        agent_params = {**computation_graph_args, **train_args}
        
        self.params = params
        self.params['agent_params'] = agent_params

        #############
        ## INIT
        #############
        # Get params, create logger
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Initialize environment and agent
        self.init_env()
        self.init_agent()

        # Maximum length for episodes
        self.params['ep_len'] = self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # simulation timestep, will be used for video saving
        self.fps = 10

    def init_env(self):
        """Load environment with fixed random seed"""
        assert self.params['env_name'] == 'Reacher-v2', (
            f"Environment {self.params['env_name']} not supported yet."
        )
        seed = self.params['seed']
        rng = np.random.RandomState(seed)
        env_seed = rng.randint(0, (1 << 31) - 1)
        self.env = ReacherWrapper(gym.make(self.params['env_name']))
        print(f"Using environment seed: {env_seed}")
        self.env.seed(int(env_seed))

    def init_agent(self):
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        self.agent = IRL_Agent(self.env, self.params['agent_params'])


    def training_loop(self):
        self.start_time = time.time()

        # Collect expert demonstrations and save to buffer
        demo_paths = self.collect_demo_trajectories(
            self.params['expert_policy'], self.params['demo_size'])
        self.agent.add_to_buffer(demo_paths)

        for itr in range(self.params['n_iter']):
            print("\n********** Iteration {} ************".format(itr))

            # Change log booleans for this iteration
            self.check_log(itr)

            reward_logs, policy_logs = self.agent.train()
            # reward_logs, policy_logs = {'Reward/loss': 0}, {'Policy/loss': 0}

            # log/save
            if self.log_video or self.logmetrics:
                model_dir = os.path.join(self.params['logdir'], f"models_itr_{itr:02d}")
                if not (os.path.exists(model_dir)):
                    os.makedirs(model_dir, exist_ok=True)
                # self.agent.reward.save(os.path.join(model_dir, "reward.pt"))
                # self.agent.actor.save(os.path.join(model_dir, "SAC"))

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, self.agent.policy, reward_logs, policy_logs)


    def check_log(self, itr):
        """Decide if we need to log metrics in this iteration"""
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


    def collect_demo_trajectories(
        self, 
        expert_policy: str, 
        batch_size: int 
    ):
        """
        :param expert_policy:  relative path to saved expert policy
        :return:
            paths: a list of trajectories
        """
        from stable_baselines3 import SAC
        expert_policy = SAC.load(expert_policy)
        print('\nRunning expert policy to collect demonstrations...')
        demo_paths = utils.sample_trajectories(
            self.env, expert_policy, batch_size
        )
        utils.check_demo_performance(demo_paths)
        return demo_paths



    def perform_logging(self, itr, eval_policy, reward_logs, policy_logs):

        last_log = reward_logs

        #######################

        # Evaluate the agent policy in true environment
        print("\nCollecting data for eval...")
        eval_paths = utils.sample_trajectories(
            self.env, eval_policy, 
            self.params['eval_batch_size']
        )  

        # paths_replay_buffer, log_probs_replay_buffer = self.agent.eval_on_replay_buffer(
        #     self.env, eval_policy, self.params['eval_batch_size']
        # )


        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            eval_video_paths = utils.sample_trajectories(
                self.env, eval_policy, MAX_NVIDEO, render=True)

            #save train/eval videos
            print('\nSaving eval rollouts as videos...')
            self.logger.log_paths_as_videos(
                eval_video_paths, itr, fps=self.fps, 
                max_videos_to_save=MAX_NVIDEO, video_title='eval_rollouts')

        #######################

        # save eval metrics
        # TODO: should parse the reward training loss and policy training loss
        # TODO: should add a visualization tool to check the trained reward function
        if self.logmetrics:
            # returns, for logging
            eval_returns = [path.rewards.sum() for path in eval_paths]
            # replay_buffer_returns = [path.rewards.sum() for path in paths_replay_buffer]

            # episode lengths, for logging
            eval_ep_lens = [len(path) for path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval/AverageReturn"] = np.mean(eval_returns)
            logs["Eval/StdReturn"] = np.std(eval_returns)
            logs["Eval/MaxReturn"] = np.max(eval_returns)
            logs["Eval/MinReturn"] = np.min(eval_returns)
            logs["Eval/AverageEpLen"] = np.mean(eval_ep_lens)

            # logs["ReplayBuffer/AverageReturn"] = np.mean(replay_buffer_returns)
            # logs["ReplayBuffer/StdReturn"] = np.std(replay_buffer_returns)
            # logs["ReplayBuffer/MaxReturn"] = np.max(replay_buffer_returns)
            # logs["ReplayBuffer/MinReturn"] = np.min(replay_buffer_returns)

            # logs["ReplayBuffer/Mean_logprob"] = np.mean(log_probs_replay_buffer)
            # logs["ReplayBuffer/Std_logprob"] = np.std(log_probs_replay_buffer)
            # logs["ReplayBuffer/Max_logprob"] = np.max(log_probs_replay_buffer)
            # logs["ReplayBuffer/Min_logprob"] = np.min(log_probs_replay_buffer)

            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(reward_logs)
            logs.update(policy_logs)

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Reacher-v2')
    parser.add_argument('--expert_policy', type=str, default='SAC_Reacher-v2')        
    parser.add_argument(
        '--n_iter', '-n', type=int, default=100,
        help='Number of total iterations')
    parser.add_argument(
        '--demo_size', type=int, default=64, 
        help='Number of expert paths to add to replay buffer'
    )
    parser.add_argument(
        '--transitions_per_itr', type=int, default=2,
        help='Number of expert transitions to sample per iteration'
    )
    parser.add_argument(
        '--reward_updates_per_itr', type=int, default=8,
        help='Number of reward updates per iteration'
    )
    parser.add_argument(
        '--agent_actions_per_demo_transition', type=int, default=4,
        help='Number of agent actions sampled for each expert_transition'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=64,
        help='Number of policy rollouts for evaluation'
    )

    parser.add_argument('--discount', type=float, default=1.0)        
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true')
    
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################
    trainer = Trainer(params)
    trainer.training_loop()
    # try:
    #     trainer = Trainer(params)
    #     trainer.training_loop()
    # except KeyboardInterrupt:
    #     keep = input("\nExiting from training early.\nKeep logs & models? ([y]/n)? ")
    #     if keep.lower() in ["n", "no"]:
    #         import shutil
    #         shutil.rmtree(logdir)

if __name__ == '__main__':
    # Allow CUDA in multiprocessing
    # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # torch.multiprocessing.set_start_method('spawn', force=True)
    
    main()