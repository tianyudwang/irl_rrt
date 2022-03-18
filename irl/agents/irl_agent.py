"""
Pytorch implementation of Guided cost learning with GAN formulation
See https://arxiv.org/pdf/1611.03852.pdf

"""
from typing import Any, Dict
from tqdm import tqdm
import time  
from collections import OrderedDict

import gym
import numpy as np

import torch as th
from torch import nn
from torch import optim

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from irl.agents.cost_net import CostNet
from irl.utils.replay_buffer import ReplayBuffer
from irl.utils import utils, wrappers
import irl.utils.pytorch_utils as ptu
import irl.planners.geometric_planner as gp 
import irl.utils.planner_utils as pu

class IRL_Agent:
    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self.params = params
        ptu.init_gpu(use_gpu=not self.params['no_gpu'], gpu_id=self.params['which_gpu'])

        self.init_env()
        self.max_episode_steps = self.env.unwrapped.spec.max_episode_steps

        self.logger = configure(self.params['logdir'], ["stdout", "tensorboard"])

        # Set up discriminator
        self.params['ob_dim'] = self.env.observation_space.shape[0]
        self.params['ac_dim'] = self.env.action_space.shape[0]
        self.cost_net = CostNet(self.params, self.logger)

        # Set up generator
        self.irl_env = wrappers.IRLWrapper(self.env, self.cost_net.reward)
        
        self.policy = SAC(
            "MlpPolicy", 
            self.irl_env, 
            batch_size=self.max_episode_steps*4, 
            learning_rate=self.params['learning_rate'],
            verbose=0
        )
        self.policy.set_logger(self.logger)

        # Set up replay buffers
        self.demo_buffer = ReplayBuffer()
        self.agent_buffer = ReplayBuffer()

    def init_env(self):
        """Load environment with fixed random seed"""
        assert self.params['env_name'] == 'Reacher-v2', (
            f"Environment {self.params['env_name']} not supported yet."
        )
        seed = self.params['seed']
        rng = np.random.RandomState(seed)
        env_seed = rng.randint(0, (1 << 31) - 1)
        self.env = wrappers.ReacherWrapper(gym.make(self.params['env_name']))
        print(f"Using environment seed: {env_seed}")
        self.env.seed(int(env_seed))


    def train(self):
        # Run expert policy to collect demonstration paths
        demo_paths = utils.collect_demo_trajectories(
            self.env,
            self.params['expert_policy'], 
            self.params['demo_size']
        )
        self.demo_buffer.add_rollouts(demo_paths)

        for itr in tqdm(range(self.params["n_iter"])):
            start_time = time.time()
            self.train_policy()
            gen_time = time.time()
            self.logger.record('time/policy', gen_time - start_time)
            self.train_reward()
            self.logger.record('time/reward', time.time() - gen_time)

            # Evaluation
            self.perform_logging(self.policy)
            self.logger.dump(itr)


    def train_reward(self):

        batch_size = self.params['reward_update_batch_size']

        # Build PRM roadmap
        planner = gp.ReacherPRMstarPlanner()
        self.cost_net.copy_model_to_cpu()

        agent_paths = self.agent_buffer.sample_recent_trajectories(batch_size)
        agent_s0 = ptu.from_numpy(np.stack([path.states[0] for path in agent_paths]))
        agent_s1 = ptu.from_numpy(np.stack([path.states[1] for path in agent_paths]))
        agent_states = pu.plan_from_states(
            planner, 
            agent_s1, 
            self.cost_net.cost, 
            solveTime=0.20
        )
        agent_states = pu.add_states_to_paths(agent_s0, agent_states)
        agent_states = pu.fixed_horizon_paths(agent_states, self.max_episode_steps)
        agent_log_probs = ptu.from_numpy(
            np.stack([path.log_probs[0] for path in agent_paths])
        ).reshape(-1,1)        

        for i in range(self.params['n_reward_updates_per_itr']):
            demo_paths = self.demo_buffer.sample_random_trajectories(batch_size)            
            demo_s0 = ptu.from_numpy(np.stack([path.states[0] for path in demo_paths]))
            demo_s1 = ptu.from_numpy(np.stack([path.states[1] for path in demo_paths]))
            demo_states = pu.plan_from_states(
                planner, 
                demo_s1, 
                self.cost_net.cost, 
                solveTime=0.05
            )
            demo_states = pu.add_states_to_paths(demo_s0, demo_states)
            demo_states = pu.fixed_horizon_paths(demo_states, self.max_episode_steps)            

            if self.params['agent_action_from_demo_state']:
                agent_s0 = demo_s0
            else:
                agent_paths = self.agent_buffer.sample_recent_trajectories(batch_size)
                agent_s0 = ptu.from_numpy(np.stack([path.states[0] for path in agent_paths]))
            
            agent_a0, agent_log_probs = utils.sample_agent_action_log_prob(
                agent_s0,
                self.policy
            )
            agent_s1 = pu.next_states_from_env(self.env, agent_s0, agent_a0)
            agent_states = pu.plan_from_states(
                planner, 
                agent_s1, 
                self.cost_net.cost, 
                solveTime=0.05
            )
            agent_states = pu.add_states_to_paths(agent_s0, agent_states)
            agent_states = pu.fixed_horizon_paths(agent_states, self.max_episode_steps)
            agent_log_probs = agent_log_probs.reshape(-1,1)

            self.cost_net.train_irl(demo_states, agent_states, agent_log_probs)


    def train_policy(self):
        """
        Train the policy/actor using learned reward
        """
        batch_size = self.params['policy_update_batch_size']
        self.policy.learn(total_timesteps=self.max_episode_steps*batch_size, log_interval=10)

        # Move rollouts to agent replay buffer
        agent_paths = utils.sample_trajectories(self.irl_env, self.policy, batch_size)
        self.agent_buffer.add_rollouts(agent_paths)

    def perform_logging(self, eval_policy):

        #######################
        # Evaluate the agent policy in true environment
        print("\nCollecting data for eval...")
        eval_paths = utils.sample_trajectories(
            self.env, eval_policy, 
            self.params['eval_batch_size']
        )  

        eval_returns = [path.rewards.sum() for path in eval_paths]
        eval_ep_lens = [len(path) for path in eval_paths]

        logs = OrderedDict()
        logs["Eval/AverageReturn"] = np.mean(eval_returns)
        logs["Eval/StdReturn"] = np.std(eval_returns)
        logs["Eval/MaxReturn"] = np.max(eval_returns)
        logs["Eval/MinReturn"] = np.min(eval_returns)
        logs["Eval/AverageEpLen"] = np.mean(eval_ep_lens)

        for key, value in logs.items():
            self.logger.record(key, value)



