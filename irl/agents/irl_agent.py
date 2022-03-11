from typing import Optional, List, Dict, Union, Tuple
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import torch as th
from torch import nn
from torch import optim
import torch.multiprocessing as mp 

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.torch_layers import FlattenExtractor

from irl.agents.base_agent import BaseAgent 
from irl.rewards.reward_net import RewardNet
import irl.planners.geometric_planner as gp 

import irl.utils.planner_utils as pu
import irl.utils.pytorch_utils as ptu 
from irl.utils import utils, types
from irl.utils.wrappers import IRLEnv, ReacherWrapper
from irl.utils.replay_buffer import ReplayBuffer


class IRL_Agent(BaseAgent):
    def __init__(
        self, 
        params: Dict[str, any]
    ) -> None:
        super().__init__()

        # init vars
        self.params = params
        ptu.init_gpu(use_gpu=not self.params['no_gpu'], gpu_id=self.params['which_gpu'])

        self.init_env()
        self.max_episode_steps = self.env.unwrapped.spec.max_episode_steps
        self.logger = configure(self.params['logdir'], ["stdout", "tensorboard"])

        # reward function
        self.reward = RewardNet(self.params, self.logger)
        
        # create a wrapper env with learned reward
        self.irl_env = IRLEnv(self.env, self.reward)

        # actor/policy with wrapped env
        self.policy = SAC("MlpPolicy", self.irl_env, verbose=1)
        self.policy.set_logger(self.logger)
        # self.policy = self.make_actor()

        # Replay buffer to hold demo transitions
        self.demo_buffer = ReplayBuffer()

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

        self.params['ob_dim'] = self.env.observation_space.shape[0]


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
        lr_schedule = get_schedule_fn(self.params['learning_rate'])
        policy.optimizer = optim.Adam(
            policy.parameters(), 
            lr=lr_schedule(1)
        )
        return policy

    def train(self):
        # Collect expert demonstrations and save to buffer
        demo_paths = self.collect_demo_trajectories(
            self.params['expert_policy'], self.params['demo_size'])
        self.demo_buffer.add_rollouts(demo_paths)

        for itr in tqdm(range(self.params['n_iter'])):    
            # Sample expert transitions from replay buffer
            demo_transitions = self.demo_buffer.sample_random_transitions(
                self.params['transitions_per_itr']
            )
            demo_states = ptu.from_numpy(
                np.stack([transition.state for transition in demo_transitions])
            )
            demo_next_states = ptu.from_numpy(
                np.stack([transition.next_state for transition in demo_transitions])
            )

            # Copy reward NN weight from cuda to cpu, and set up PRM planner
            # set model to eval mode in case there are BatchNorm, Dropout layers
            self.reward.copy_model_to_cpu()
            self.reward.model_cpu.eval()
            planner = gp.ReacherPRMstarPlanner()

            # Plan expert paths from expert next states
            demo_paths = pu.plan_from_states(
                planner, 
                demo_next_states, 
                self.reward.cost_fn, 
                solveTime=0.2
            )
            # Add first state back to each path
            demo_paths = pu.add_states_to_paths(demo_states, demo_paths)
            demo_paths = pu.fixed_horizon_paths(demo_paths, self.max_episode_steps)

            # Optimize reward
            for i in range(self.params['reward_updates_per_itr']):
                # Sample agent actions with log probs and plan from agent next states
                agent_actions, agent_log_probs = utils.sample_agent_action_log_prob(
                    demo_states,
                    self.policy
                )
                agent_next_states = pu.next_states_from_env(self.env, demo_states, agent_actions)
                agent_paths = pu.plan_from_states(
                    planner, 
                    agent_next_states, 
                    self.reward.cost_fn, 
                    solveTime=0.02
                )
                agent_paths = pu.add_states_to_paths(demo_states, agent_paths)
                agent_paths = pu.fixed_horizon_paths(agent_paths, self.max_episode_steps)

                self.reward.update(demo_paths, agent_paths, agent_log_probs, itr)

                # # Multiple agent actions sampled per expert transition
                # # Sample agent actions with log probs and plan from agent next states
                # agent_paths_l, agent_log_probs_l = [], []
                # for j in range(self.params['agent_actions_per_demo_transition']):
                #     agent_actions, agent_log_probs = utils.sample_agent_action_log_prob(
                #         demo_states,
                #         self.policy
                #     )
                #     agent_next_states = pu.next_states_from_env(self.env, demo_states, agent_actions)
                #     agent_paths = pu.plan_from_states(
                #         planner, 
                #         agent_next_states, 
                #         self.reward.cost_fn, 
                #         solveTime=0.02
                #     )
                #     agent_paths = pu.add_states_to_paths(demo_states, agent_paths)
                #     agent_paths_l.append(agent_paths)
                #     agent_log_probs_l.append(agent_log_probs)
                # self.reward.update(demo_paths, agent_paths_l, agent_log_probs_l, itr)

            # Optimize policy
            # policy_logs = self.train_policy(agent_paths_l, agent_log_probs_l)
            self.train_policy()

            # Perform logging
            print('\nBeginning logging procedure...')
            self.perform_logging(itr, self.policy)
            self.logger.dump(itr)

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
        batch_size = self.params['policy_update_batch_size']
        self.policy.learn(total_timesteps=self.max_episode_steps*batch_size, log_interval=10)


    #####################################################
    #####################################################
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

    def perform_logging(self, itr, eval_policy):

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