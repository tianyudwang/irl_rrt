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
        self.policy = SAC("MlpPolicy", self.env, verbose=1)
        # self.policy = self.make_actor()

        self.state_dim = self.agent_params['ob_dim']

        # self.planner = PendulumSSTPlanner()
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

        # # Sample agent actions from expert states and compute next states
        agent_actions_l, agent_log_probs_l = self.sample_agent_action_log_prob(demo_states)
        agent_next_states_l = self.next_states_from_env(demo_states, agent_actions_l)

        # Copy reward NN weight from cuda to cpu, 
        # set model to eval mode in case there are BatchNorm, Dropout layers
        # and update to planner
        self.reward.copy_model_to_cpu()
        self.reward.model_cpu.eval()

        # Plan optimal paths from next states to goal under current reward function        
        demo_paths = planner_utils.plan_from_states(demo_next_states, self.reward.cost_fn)
        agent_paths_l = [
            planner_utils.plan_from_states(agent_next_states, self.reward.cost_fn) 
            for agent_next_states in agent_next_states_l
        ]

        # Add first state back to each path
        demo_paths = planner_utils.add_states_to_paths(demo_states, demo_paths)
        agent_paths_l = [
            planner_utils.add_states_to_paths(demo_states, agent_paths)
            for agent_paths in agent_paths_l
        ]

        # Optimize reward
        reward_logs = self.reward.update(demo_paths, agent_paths_l, agent_log_probs_l)
        # reward_logs = {"Reward/loss": 0}

        # Optimize policy
        # policy_logs = self.train_policy(agent_paths_l, agent_log_probs_l)
        policy_logs = self.train_policy()

        return reward_logs, policy_logs


    def next_states_from_env(
        self, 
        states: th.Tensor, 
        actions_l: List[th.Tensor]
    ) -> List[th.Tensor]:
        """Query the environment for next states"""
        # states = ptu.to_numpy(states)
        # actions = ptu.to_numpy(actions)
        # next_states = []
        # for state, action in zip(states, actions):
        #     next_states.append(self.env.one_step_transition(state, action))
        # return ptu.from_numpy(np.stack(next_states))

        states = ptu.to_numpy(states)
        actions_l = [ptu.to_numpy(actions) for actions in actions_l]
        next_states_l = []
        for actions in actions_l:
            assert len(states) == len(actions), "Sampled actions not equal to states"
            next_states = []
            for state, action in zip(states, actions):
                next_states.append(self.env.one_step_transition(state, action))
            next_states = ptu.from_numpy(np.stack(next_states))
            assert next_states.shape == states.shape, "Sampled next states not equal to states"
            next_states_l.append(next_states)
        return next_states_l

    def sample_agent_action_log_prob(
        self, 
        demo_states: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Sample agent actions and log probabilities from demo states"""
        if isinstance(self.policy, SAC):
            policy = self.policy.policy.actor
        elif isinstance(self.policy, Actor): 
            policy = self.policy
        else:
            assert False, f"Policy class {type(self.policy)} not implemented"
        action_log_prob_l = [
            policy.action_log_prob(demo_states)
            for _ in range(self.agent_params['agent_actions_per_demo_transition'])
        ]
        agent_actions, agent_log_probs = list(zip(*action_log_prob_l))
        return agent_actions, agent_log_probs


    # def train_policy(
    #     self, 
    #     agent_paths_l: List[List[th.Tensor]],
    #     agent_log_probs_l: List[th.Tensor]
    # ) -> Dict[str, float]:

    #     assert len(agent_paths_l) == len(agent_log_probs_l) == self.agent_params['agent_actions_per_demo_transition']
        
    #     agent_Qs = []
    #     policy_loss = 0
    #     for agent_paths, agent_log_probs in zip(agent_paths_l, agent_log_probs_l):
    #         agent_Q = th.cat([self.reward.compute_Q(path, debug=True) for path in agent_paths])
    #         agent_Qs.append(agent_Q)
    #         policy_loss += (agent_log_probs - agent_Q).mean()

    #     self.policy.optimizer.zero_grad()
    #     policy_loss.backward()
    #     self.policy.optimizer.step()

    #     policy_logs = {
    #         "Policy/agent_Q": ptu.to_numpy(th.mean(th.cat(agent_Qs))),
    #         "Policy/agent_log_prob": ptu.to_numpy(th.mean(th.cat(agent_log_probs_l))),
    #         "Policy/loss": ptu.to_numpy(policy_loss)
    #     }

    #     for loss_name, loss_val in policy_logs.items():
    #         print(loss_name, f"{loss_val.item():.2f}")
    #     return policy_logs


    def train_policy(self):
        """
        Train the policy/actor using learned reward
        """
        print('\nTraining agent policy...')
        self.policy.learn(total_timesteps=1024*4, log_interval=16)
        train_log = {'Policy/policy_loss': 0}
        return train_log

    #####################################################
    #####################################################
    
    def add_to_buffer(self, paths: List[np.ndarray]) -> None:
        self.demo_buffer.add_rollouts(paths)

    def sample_transitions(self, batch_size: int) -> List[types.Transition]:
        return self.demo_buffer.sample_random_transitions(batch_size)


    def eval_on_replay_buffer(
        self, 
        env,
        policy,
        batch_size: Optional[int] = 64
    ) -> Tuple[List[types.Trajectory], np.ndarray]:
        """Evaluate the policy by setting initial states from replay buffer
        Also check the action log prob"""
        demo_transitions = self.sample_transitions(batch_size)

        paths = []
        for transition in demo_transitions:
            qpos = transition.info['qpos']
            qvel = transition.info['qvel']
            paths.append(utils.sample_trajectory(env, policy, qpos, qvel))
            
        # Compute action log prob on all transitions 
        assert isinstance(policy, Actor)
        log_probs = []
        transitions = self.demo_buffer.transitions
        # transitions = self.sample_transitions(128)
        for transition in transitions:
            obs, _ = policy.obs_to_tensor(transition.state)
            mean_actions, log_std, kwargs = policy.get_action_dist_params(obs)
            action_dist = policy.action_dist.proba_distribution(mean_actions, log_std)
            action = ptu.from_numpy(transition.action).reshape(1, -1)
            log_prob = action_dist.log_prob(action)
            log_prob = np.clip(ptu.to_numpy(log_prob)[0], -20, 10)
            log_probs.append(log_prob)

        return paths, np.array(log_probs)

