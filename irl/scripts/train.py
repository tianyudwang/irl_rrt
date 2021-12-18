from typing import List
import argparse
import os
import time
from collections import OrderedDict

import gym
import numpy as np 
import torch 

from irl.agents.irl_agent import IRL_Agent
import irl.util.pytorch_util as ptu 
import irl.util.utils as utils
from irl.util.logger import Logger
from irl.util.wrappers import PendulumWrapper


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
            'reward_updates_per_iter': params['reward_updates_per_iter'],
            'transitions_per_reward_update': params['transitions_per_reward_update'],
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
        assert self.params['env_name'] == 'Pendulum-v0', f"Environment {self.params['env_name']} not supported yet."
        seed = self.params['seed']
        rng = np.random.RandomState(seed)
        env_seed = rng.randint(0, (1 << 31) - 1)
        self.env = PendulumWrapper(gym.make("Pendulum-v0"))
        print(env_seed)
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

        # Collect expert demonstrations
        demo_paths = self.collect_demo_trajectories(
            self.params['expert_policy'], self.params['demo_size'])
        self.agent.add_to_buffer(demo_paths)

        for itr in range(self.params['n_iter']):
            print("\n********** Iteration {} ************".format(itr))

            # decide if videos should be rendered/logged at this iteration
            if ((itr+1) % self.params['video_log_freq'] == 0 
                and self.params['video_log_freq'] != -1):
                self.log_video = True
            else:
                self.log_video = False
            # decide if metrics should be logged
            if ((itr+1) % self.params['scalar_log_freq'] == 0 
                and self.params['scalar_log_freq'] != -1):
                self.logmetrics = True
            else:
                self.logmetrics = False

            # Collect agent trajectories
            agent_paths, train_video_paths = self.collect_agent_trajectories(
                self.agent.actor, self.params['demo_size'])

            reward_logs = self.agent.train_reward()

            for step in range(self.params['policy_updates_per_iter']):
                policy_logs = self.agent.train_policy()

            # log/save
            self.save_model(itr)
            if self.log_video or self.logmetrics:
                self.agent.actor.save("../models/SAC_NavEnv-v0_itr_{}".format(itr))
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, agent_paths, self.agent.actor, 
                    train_video_paths, reward_logs, policy_logs)
                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

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
            self.env, expert_policy, batch_size)
        #demo_paths = utils.pad_absorbing_states(demo_paths)
        utils.check_demo_performance(demo_paths)
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
        paths = utils.sample_trajectories(
            self.env, collect_policy, batch_size)
        #paths = utils.pad_absorbing_states(paths)

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_trajectories(
                self.env, collect_policy, MAX_NVIDEO, render=True)
        return paths, train_video_paths


    def save_model(self, itr: int) -> None:
        # Save model to local        
        model_dir = os.path.join(self.params['logdir'], 'model')
        if not(os.path.exists(model_dir)):
            os.makedirs(model_dir)
        self.agent.save_reward_model(os.path.join(model_dir, f'itr_{itr:02d}.pt'))      

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, 
                        reward_logs, policy_logs):

        last_log = reward_logs[-1]

        #######################

        # Evaluate the agent policy in true environment
        print("\nCollecting data for eval...")
        eval_paths = utils.sample_trajectories(
            self.env, eval_policy, 
            self.params['eval_batch_size'], render=False
        )  

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            eval_video_paths = utils.sample_trajectories(
                self.env, eval_policy, MAX_NVIDEO, render=True)

            #save train/eval videos
            print('\nSaving train and eval rollouts as videos...')
            self.logger.log_paths_as_videos(
                train_video_paths, itr, fps=self.fps, 
                max_videos_to_save=MAX_NVIDEO, video_title='train_rollouts')
            self.logger.log_paths_as_videos(
                eval_video_paths, itr, fps=self.fps, 
                max_videos_to_save=MAX_NVIDEO, video_title='eval_rollouts')

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
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pendulum-v0')
    parser.add_argument('--exp_name', type=str, default='Pendulum-v0')
    parser.add_argument('--expert_policy', type=str, default='SAC_Pendulum-v0')        
    parser.add_argument(
        '--n_iter', '-n', type=int, default=100,
        help='Number of total iterations')
    parser.add_argument(
        '--demo_size', type=int, default=10, 
        help='Number of expert paths to add to replay buffer'
    )
    parser.add_argument(
        '--reward_updates_per_iter', type=int, default=8,
        help='Number of reward updates per iteration'
    )
    parser.add_argument(
        '--policy_updates_per_iter', type=int, default=8,
        help='Number of policy updates per iteration'
    )
    parser.add_argument(
        '--transitions_per_reward_update', type=int, default=128,
        help='Number of agent transitions per reward update'
    )
    parser.add_argument(
        '--agent_actions_per_demo_transition', type=int, default=1,
        help='Number of agent actions sampled for each expert_transition'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=10,
        help='Number of policy rollouts for evaluation'
    )

    parser.add_argument('--discount', type=float, default=1.0)        
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
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

if __name__ == '__main__':
    # Allow CUDA in multiprocessing
    # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    main()