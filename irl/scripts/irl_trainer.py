import time
from collections import OrderedDict

import gym
import numpy as np 
import torch

from irl.agents.mlp_reward import MLPReward
import irl.scripts.pytorch_util as ptu 
import irl.scripts.utils as utils
from irl.scripts.logger import Logger

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

# Custom env wrappers to change reward function
class PendulumEnv(gym.Wrapper):
    def __init__(self, env, reward):
        """
        Override the true environment reward with the NN reward
        """
        gym.Wrapper.__init__(self, env)
        self.reward = reward 

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(
            ptu.from_numpy(obs).reshape(1, -1), 
            ptu.from_numpy(action).reshape(1, -1)
        ).item()
        
        # Check whether Pendulum-v0 is inverted
        # obs: Box(3), cos(theta) [-1.0, 1.0], sin(theta) [-1.0, 1.0], theta dot [-8.0, 8.0]
#        if abs(obs[0] - 1) < 0.01 and abs(obs[1]) < 0.02 and abs(obs[2]) < 0.01:
#            done = True or done
#        else:
#            done = False or done
        # TODO: check if performance changes when first exit or finite horizon
        return obs, reward, done, info

class FetchReachEnv(gym.Wrapper):
    def __init__(self, env, reward):
        """
        Override the true environment reward with the NN reward
        """
        gym.Wrapper.__init__(self, env)
        self.reward = reward 

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(
            ptu.from_numpy(obs).reshape(1, -1), 
            ptu.from_numpy(action).reshape(1, -1)
        ).item()
        return obs, reward, done, info

class IRL_Trainer():

    def __init__(self, params):

        #############
        ## INIT
        #############
        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        if self.params['env_name'] == 'Pendulum-v0':
            self.env = gym.make('Pendulum-v0')
        elif self.params['env_name'] == 'FetchReach-v1':
            from gym.wrappers import FilterObservation, FlattenObservation
            env = gym.make('FetchReach-v1', reward_type='dense')
            self.env = FlattenObservation(
                FilterObservation(
                    env, 
                    filter_keys=['observation', 'desired_goal']
                )
            )
        else:
            raise ValueError('Environment not supported')
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # simulation timestep, will be used for video saving
        self.fps = 10

        #############
        ## AGENT
        #############

        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2
        #import pdb; pdb.set_trace()
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # Initialize reward neural network
        self.reward = MLPReward(
            self.params['agent_params']['ac_dim'],
            self.params['agent_params']['ob_dim'],
            self.params['agent_params']['n_layers'],
            self.params['agent_params']['size'],
            self.params['agent_params']['output_size'],
            learning_rate=self.params['agent_params']['learning_rate']
        )

        # Make IRL env wrappers with learned reward
        if self.params['env_name'] == 'Pendulum-v0':
            self.irl_env = PendulumEnv(self.env, self.reward)
        elif self.params['env_name'] == 'FetchReach-v1':
            self.irl_env = FetchReachEnv(self.env, self.reward)

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.irl_env, self.reward, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy, expert_policy=None):
        """
        :param n_iter:  number of iterations
        :param collect_policy: 
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        
        # add demonstrations to replay buffer
        demo_paths = self.collect_demo_trajectories(expert_policy)
        self.agent.add_to_buffer(demo_paths, demo=True)

        for itr in range(n_iter):
            print("\n********** Iteration {} ************".format(itr))

            # TODO: set up logging
            # decide if videos should be rendered/logged at this iteration
            if (itr+1) % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False
            # decide if metrics should be logged
            if (itr+1) % self.params['scalar_log_freq'] == 0 and self.params['scalar_log_freq'] != -1:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # Generate agent paths from current agent policy
            agent_paths, envsteps_this_batch, train_video_paths = self.collect_training_trajectories(
                collect_policy, self.params['sample_size']
            )
            self.agent.add_to_buffer(paths)
            self.total_envsteps += envsteps_this_batch

            # Train reward
            reward_logs = self.train_reward()

            # Train policy using any policy optimization method
            policy_logs = self.train_policy()

            # log/save
            if self.log_video or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, agent_paths, eval_policy, train_video_paths, reward_logs, policy_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    def collect_demo_trajectories(self, expert_policy):
        """
        :param expert_policy:  relative path to saved expert policy
        :return:
            paths: a list of trajectories
        """
        from stable_baselines3 import SAC
        expert_policy = SAC.load(expert_policy)
        print('\nRunning expert policy to collect demonstrations...')
        demo_paths, _ = utils.sample_trajectories(
            self.env, 
            expert_policy, 
            batch_size=self.params['demo_size'], 
            expert=True
        )
        return demo_paths

    def collect_training_trajectories(self, collect_policy, batch_size):
        """
        :param collect_policy:  the current policy which we use to collect data
        :param batch_size:  the number of trajectories to collect
        :return:
            paths: a list trajectories
            train_video_paths: paths which also contain videos for visualization purposes
        """
        print("\nCollecting sample trajectories to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env, collect_policy, batch_size
        )

        # TODO: add logging and training videos
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths, _ = utils.sample_trajectories(
                self.env, collect_policy, MAX_NVIDEO, render=True
            )
        return paths, envsteps_this_batch, train_video_paths

    def train_reward(self):
        """
        IRL with RRT*
        """
        print("\nUpdating reward parameters...")
        reward_logs = []
        for k in range(self.params['reward_updates_per_iter']):
            # Use the sampled data to train the reward function
            reward_log = self.agent.train_reward()
            reward_logs.append(reward_log)
        return reward_logs

    def train_policy(self):
        """
        Use SB3 SAC policy training using learned reward as black box
        """
        print('\nTraining agent using learned reward...')
        train_logs = []
        for train_step in range(self.params['policy_updates_per_iter']):
            train_logs.append(self.agent.train_policy())
        return train_logs

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, reward_logs, policy_logs):

        last_log = reward_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, _ = utils.sample_trajectories(
            self.env, eval_policy, 
            self.params['eval_batch_size'], render=False
        )

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            eval_video_paths, _ = utils.sample_trajectories(self.env, eval_policy, MAX_NVIDEO, render=True)

            #save train/eval videos
            print('\nSaving train and eval rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

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

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

