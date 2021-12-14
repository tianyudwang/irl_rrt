from typing import Optional, List

import numpy as np
import gym
from stable_baselines3 import SAC

from irl.agents.base_agent import BaseAgent 
from irl.rewards.reward_net import RewardNet
from irl.planners.sst_planner import SSTPlanner

import irl.util.pytorch_util as ptu 
import irl.util.utils as utils
from irl.util.wrappers import IRLEnv
from irl.util.replay_buffer import ReplayBuffer


class IRL_Agent(BaseAgent):
    def __init__(self, env, agent_params):
        super(IRL_Agent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # reward function
        self.reward = RewardNet(self.agent_params)
        
        # create a wrapper env with learned reward
        self.irl_env = IRLEnv(self.env, self.reward)

        # actor/policy with wrapped env
        self.actor = SAC("MlpPolicy", self.irl_env, verbose=1)

        self.state_dim = self.agent_params['ob_dim']

        self.planner = SSTPlanner()

        # Replay buffer to hold demo transitions (maximum transitions)
        self.demo_buffer = ReplayBuffer(10000)

    def train_reward(self):
        """
        Train the reward function
        """
        print('\nTraining agent reward function...')
        demo_transitions = self.sample_transitions(self.agent_params['transitions_per_reward_update'])

        # Update OMPL SimpleSetup object cost function with current learned reward
        self.planner.update_ss_cost(self.reward.cost_fn)

        # Synchronize reward net weights on cpu and cuda 
        self.reward.copy_model_to_cpu()

        demo_paths = []
        agent_paths = []
        agent_log_probs = []

        for i in range(self.agent_params['transitions_per_reward_update']):
            # Sample expert transitions (s, a, s')
            # and find optimal path from s' to goal
            print('Planning trajectory from expert state ...')
            ob, ac, log_probs, rewards, next_ob, done = [var[i] for var in demo_transitions]
            path, controls = self.planner.plan(next_ob)
            path = np.concatenate((ob.reshape(1, self.state_dim), path), axis=0)
            demo_paths.append([path])
            
            # and find optimal path from s'_a to goal
            paths = []
            log_probs = []
            for j in range(self.agent_params['agent_actions_per_demo_transition']):
                # Sample agent transitions (s, a, s') at each expert state s
                print(f"Planning trajectory {j+1}/{self.agent_params['agent_actions_per_demo_transition']} from agent state")
                agent_ac, _ = self.actor.predict(ob)
                log_prob = utils.get_log_prob(self.actor, agent_ac)
                agent_next_ob = self.env.one_step_transition(ob, agent_ac)

                
                # Find optimal path from s' to goal
                path, controls = self.planner.plan(agent_next_ob)
                path = np.concatenate((ob.reshape(1, self.state_dim), path), axis=0)
                paths.append(path)
                log_probs.append(log_prob)
            agent_paths.append(paths)
            agent_log_probs.append(log_probs)

        demo_paths = self.collate_fn(demo_paths)
        agent_paths = self.collate_fn(agent_paths)
        agent_log_probs = np.array(agent_log_probs)

        reward_logs = []
        for step in range(self.agent_params['reward_updates_per_iter']):
            reward_logs.append(self.reward.update(demo_paths, agent_paths, agent_log_probs))
        return reward_logs

    def collate_fn(self, paths):
        """
        Pad the list of variable-length paths with goal locations
        """
        T = max([len(p) for path_l in paths for p in path_l])
        paths = np.array([[np.pad(p, ((0, T-p.shape[0]),(0,0)), 'edge') 
                 for p in path_l] for path_l in paths])
        return paths

#    def plan_optimal_paths(self, transitions):
#        """
#        For each transition (s, a, s'), we find the optimal path from s' to goal
#        """
#        num_transitions = transitions[0].shape[0]
#        paths = []
#        for i in range(num_transitions):
#            obs, ac, rew, next_obs, done = [var[i] for var in transitions]
#            path = self.RRT_plan(next_obs)
#            paths.append(path)
#        return paths


    def train_policy(self):
        """
        Train the policy/actor using learned reward
        """
        print('\nTraining agent policy...')
        self.actor.learn(total_timesteps=1000, log_interval=5)
        train_log = {'Policy loss': 0}
        return train_log

    #####################################################
    #####################################################
    
    def add_to_buffer(self, paths: List[np.ndarray]) -> None:
        """
        Add paths to demo buffer
        """
        self.demo_buffer.add_rollouts(paths)

    def sample_rollouts(self, batch_size: int) -> List[np.ndarray]:
        """
        Sample paths from demo buffer
        """
        return self.demo_buffer.sample_recent_rollouts(batch_size)

    def sample_transitions(self, batch_size: int) -> List[np.ndarray]:
        """
        Sample transitions from demo buffer
        returns observations, actions, rewards, next_observations, terminals
        """
        return self.demo_buffer.sample_random_data(batch_size)

