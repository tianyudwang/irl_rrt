from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from torch import optim
import torch.multiprocessing as mp 

import gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.torch_layers import FlattenExtractor

from irl.agents.base_agent import BaseAgent 
from irl.rewards.reward_net import RewardNet
from irl.planners.control_planner import PendulumSSTPlanner
# from irl.planners.planner import Planner

import irl.utils.pytorch_util as ptu 
from irl.utils import utils, types, planner_utils
from irl.utils.wrappers import IRLEnv
from irl.utils.replay_buffer import ReplayBuffer


class IRL_Agent(BaseAgent):
    def __init__(
        self, 
        env: gym.Env, 
        agent_params: Dict[str, any]
    ) -> None:
        super(IRL_Agent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # reward function
        self.reward = RewardNet(self.agent_params)
        
        # create a wrapper env with learned reward
        self.irl_env = IRLEnv(self.env, self.reward)

        # actor/policy with wrapped env
        # self.actor = SAC("MlpPolicy", self.irl_env, verbose=1)
        self.policy = self.make_actor()

        self.state_dim = self.agent_params['ob_dim']

        self.planner = PendulumSSTPlanner()
        # self.planner = Planner()

        # Replay buffer to hold demo transitions
        self.demo_buffer = ReplayBuffer()

    def make_actor(self):
        """Construct an actor as policy"""
        actor_kwargs = {
            "observation_space": self.irl_env.observation_space,
            "action_space": self.irl_env.action_space,
            "net_arch": [256, 256],
            "activation_fn": nn.ReLU,
            "normalize_images": False
        }
        features_extractor = FlattenExtractor(self.irl_env.observation_space)
        actor_kwargs.update(
            dict(features_extractor=features_extractor, 
                features_dim=features_extractor.features_dim)
        )
        policy = Actor(**actor_kwargs).to("cuda")

        from stable_baselines3.common.utils import get_schedule_fn
        lr_schedule = get_schedule_fn(self.agent_params['learning_rate'])
        policy.optimizer = optim.Adam(
            policy.parameters(), 
            lr=lr_schedule(1)
        )

        return policy

    def train(self):

        # Sample a minibatch of expert transitions
        demo_transitions = self.sample_transitions(self.agent_params['transitions_per_itr'])
        demo_states = ptu.from_numpy(
            np.stack([transition.state for transition in demo_transitions])
        )
        demo_next_states = ptu.from_numpy(
            np.stack([transition.next_state for transition in demo_transitions])
        )

        # Sample agent actions from expert states and compute next states
        agent_actions, agent_log_probs = self.policy.action_log_prob(demo_states)

        agent_next_states = self.add_next_states_from_env(demo_states, agent_actions)
        assert agent_next_states.shape == demo_states.shape

        # Copy reward NN weight from cuda to cpu, 
        # set model to eval mode in case there are BatchNorm, Dropout layers
        # and update to planner
        self.reward.copy_model_to_cpu()
        # self.reward.model_cpu.eval()
        # self.planner.update_ss_cost(self.reward.cost_fn)

        # Plan optimal paths from next states to goal under current reward function        
        demo_paths = planner_utils.plan_from_states(demo_next_states, self.reward.cost_fn)
        agent_paths = planner_utils.plan_from_states(agent_next_states, self.reward.cost_fn)

        # demo_paths = self.plan_from_states(demo_next_states)
        # agent_paths = self.plan_from_states(agent_next_states)

        # Add first state back to each path
        demo_paths = [
            th.cat((state.reshape(1, -1), path), dim=0) 
            for state, path in zip(demo_states, demo_paths)
        ]
        agent_paths = [
            th.cat((state.reshape(1, -1), path), dim=0)
            for state, path in zip(demo_states, agent_paths)
        ]

        # Optimize reward
        # for i in range(4):
        reward_logs = self.reward.update(demo_paths, agent_paths, agent_log_probs)
        
        # Optimize policy
        # for i in range(4):
        policy_logs = self.train_policy(agent_paths, agent_log_probs)


        return reward_logs, policy_logs


    def add_next_states_from_env(
            self, 
            states: th.Tensor, 
            actions: th.Tensor
        ) -> th.Tensor:
        """Query the environment for next states"""
        states = ptu.to_numpy(states)
        actions = ptu.to_numpy(actions)

        assert len(states) == len(actions)
        next_states = []
        for state, action in zip(states, actions):
            next_states.append(self.env.one_step_transition(state, action))

        agent_next_states = ptu.from_numpy(np.stack(next_states))
        return agent_next_states


    def train_policy(
            self, 
            agent_paths: List[th.Tensor],
            agent_log_probs: th.Tensor
        ) -> Dict[str, float]:

        agent_Q = th.cat([self.reward.compute_Q(path) for path in agent_paths])
        policy_loss = (agent_log_probs - agent_Q).mean()

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        policy_logs = {
            "Policy_loss": ptu.to_numpy(policy_loss)
        }

        output_str = ""
        for loss_name, loss_val in policy_logs.items():
            output_str += loss_name + f" {loss_val.item():.2f} " 
        print("\n", output_str)
        return policy_logs

    # def train_reward(self) -> Mapping[str, float]:
    #     """Train the reward function"""
    #     print('\nTraining agent reward function...')
    #     demo_transitions = self.sample_transitions(self.agent_params['transitions_per_reward_update'])

    #     # Synchronize reward net weights on cpu and cuda 
    #     self.reward.copy_model_to_cpu()
    #     # Update OMPL SimpleSetup object cost function with current learned reward
    #     self.reward.model_cpu.eval()
    #     self.planner.update_ss_cost(self.reward.cost_fn)

    #     demo_paths = []
    #     agent_paths = []
    #     agent_log_probs = []

    #     for i in range(self.agent_params['transitions_per_reward_update']):
    #         # Sample expert transitions (s, a, s')
    #         # and find optimal path from s' to goal
    #         print(f"Planning trajectory {i+1}/{self.agent_params['transitions_per_reward_update']} from expert state ...")
    #         ob, ac, log_probs, rewards, next_ob, done = [var[i] for var in demo_transitions]
    #         status, path, controls = self.planner.plan(next_ob)
    #         path = np.concatenate((ob.reshape(1, self.state_dim), path), axis=0)
    #         demo_paths.append([path])
            
    #         # and find optimal path from s'_a to goal
    #         paths = []
    #         log_probs = []
    #         for j in range(self.agent_params['agent_actions_per_demo_transition']):
    #             # Sample agent transitions (s, a, s') at each expert state s
    #             print(f"Planning trajectory {j+1}/{self.agent_params['agent_actions_per_demo_transition']} from agent state")
    #             # agent_ac, _ = self.actor.predict(ob)
    #             # log_prob = utils.get_log_prob(self.actor, agent_ac)
    #             agent_ac, log_prob = utils.action_log_prob(self.actor, ob)
    #             agent_next_ob = self.env.one_step_transition(ob, agent_ac)
                
    #             # Find optimal path from s' to goal
    #             status, path, controls = self.planner.plan(agent_next_ob)
    #             path = np.concatenate((ob.reshape(1, self.state_dim), path), axis=0)
    #             paths.append(path)
    #             log_probs.append(log_prob)
    #         agent_paths.append(paths)
    #         agent_log_probs.append(np.array(log_probs))

    #     # demo_paths = self.collate_fn(demo_paths)
    #     # agent_paths = self.collate_fn(agent_paths)
    #     # agent_log_probs = np.array(agent_log_probs)

    #     reward_logs = []
    #     for step in range(self.agent_params['reward_updates_per_iter']):
    #         reward_logs.append(self.reward.update(demo_paths, agent_paths, agent_log_probs))
    #     return reward_logs


    # def train_reward_mp(self) -> Mapping[str, float]:
    #     """Same function as above but uses multiprocessing to plan optimal paths"""
    #     print('\nTraining agent reward function...')
    #     demo_transitions = self.sample_transitions(self.agent_params['transitions_per_reward_update'])
    #     demo_transitions = [[var[i] for var in demo_transitions]
    #         for i in range(self.agent_params['transitions_per_reward_update'])]
    #     agent_next_states, agent_log_probs = self.sample_agent_transitions(demo_transitions)
        
    #     # Synchronize reward net weights for cost inference on cpu in motion planning
    #     self.reward.copy_model_to_cpu()
        
    #     # Update OMPL SimpleSetup object cost function with current learned reward
    #     self.planner.update_cost(self.reward.cost_fn)

    #     demo_paths, agent_paths = self.planner.plan_mp(demo_transitions, agent_next_states)

    #     reward_logs = []
    #     for step in range(self.agent_params['reward_updates_per_iter']):
    #         reward_logs.append(self.reward.update(demo_paths, agent_paths, agent_log_probs))
    #     return reward_logs

    # def sample_agent_transitions(
    #         self, 
    #         demo_transitions: List[List[np.ndarray]]
    #     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    #     """Sample agent policy transitions at each expert state"""
    #     agent_next_states = [[] for _ in range(len(demo_transitions))]
    #     agent_log_probs = [[] for _ in range(len(demo_transitions))]
    #     for i, demo_transition in enumerate(demo_transitions):
    #         state = demo_transition[0]
    #         for j in range(self.agent_params['agent_actions_per_demo_transition']):
    #             agent_ac, _ = self.actor.predict(state)
    #             log_prob = utils.get_log_prob(self.actor, agent_ac)
    #             agent_next_state = self.env.one_step_transition(state, agent_ac)
    #             agent_next_states[i].append(agent_next_state)
    #             agent_log_probs[i].append(log_prob)
    #     return agent_next_states, agent_log_probs

    # def plan_paths_from_demo_state(
    #         self, 
    #         demo_transition: List[np.ndarray]
    #     ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    #     """Plan optimal paths from expert and agent next_states"""
    #     state, _, _, _, next_state, _ = demo_transition
    #     path, controls = self.planner.plan(next_state)
    #     demo_paths = [np.concatenate((state.reshape(1, self.state_dim), path), axis=0)]

    #     # agent_paths, agent_log_probs = [], []
    #     # for j in range(self.agent_params['agent_actions_per_demo_transition']):
    #     #     agent_ac, _ = self.actor.predict(state)
    #     #     log_prob = utils.get_log_prob(self.actor, agent_ac)
    #     #     agent_next_state = self.env.one_step_transition(state, agent_ac)
    #     #     path, controls = self.planner.plan(agent_next_state)
    #     #     path = np.concatenate((state.reshape(1, self.state_dim), path), axis=0)
    #     #     agent_paths.append(path)
    #     #     agent_log_probs.append(log_prob)

    #     return demo_paths#, agent_paths, agent_log_probs


    # def collate_fn(self, paths):
    #     """
    #     Pad the list of variable-length paths with goal locations
    #     """
    #     T = max([len(p) for path_l in paths for p in path_l])
    #     paths = np.array([[np.pad(p, ((0, T-p.shape[0]),(0,0)), 'edge') 
    #              for p in path_l] for path_l in paths])
    #     return paths

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



    # def train_policy(self):
    #     """
    #     Train the policy/actor using learned reward
    #     """
    #     print('\nTraining agent policy...')
    #     self.actor.learn(total_timesteps=10000, log_interval=5)
    #     train_log = {'Policy loss': 0}
    #     return train_log

    #####################################################
    #####################################################
    
    def add_to_buffer(self, paths: List[np.ndarray]) -> None:
        self.demo_buffer.add_rollouts(paths)

    def sample_transitions(self, batch_size: int) -> List[types.Transition]:
        return self.demo_buffer.sample_random_transitions(batch_size)
