from typing import List

import numpy as np
import gym
from stable_baselines3 import SAC

from irl.agents.base_agent import BaseAgent
# from irl.planners import 
# from irl.agents.base_planner.base_planner_PointUMaze import ControlPlanner, GeometricPlanner
from irl.rewards.mlp_reward import MLPReward
from irl.utils.replay_buffer import ReplayBuffer
from irl.utils.wrappers import MazeIRLWrapper
import irl.utils.pytorch_util as ptu
import irl.utils.utils as utils
import irl.utils.types as types


class IRLAgent(BaseAgent):
    def __init__(
        self,
        env: gym.Env,
        agent_params: dict,
        plannerType: str
    ):
        super(IRLAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # reward function
        self.reward = MLPReward(
            self.agent_params["ob_dim"],
            self.agent_params["ac_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            self.agent_params["output_size"],
            self.agent_params["learning_rate"],
        )

        # create a wrapper env with learned reward
        self.irl_env = MazeIRLWrapper(self.env, self.reward)

        # actor/policy with wrapped irl env
        self.actor = SAC("MlpPolicy", self.irl_env, verbose=1, device=ptu.device)

        self.state_dim = self.agent_params["ob_dim"]

        # choose control planning or geometric planning
        # if plannerType.lower() in ["sst", "rrt"]:
        #     self.planner = ControlPlanner(self.env, plannerType, log_level=0)
        #     print(f"Using {plannerType.upper()} control planner")

        # elif:
        #     self.planner = GeometricPlanner(self.env, plannerType, log_level=0)
        #     print(f"Using {plannerType.upper()} geometric planner")
        # else:
        #     raise ValueError(f"{plannerType} not supported")
                
        # Replay buffer to hold demo transitions
        self.demo_buffer = ReplayBuffer()

    def train_reward(self):
        """Train the reward function"""
        print("\nTraining agent reward function...")
        demo_transitions = self.sample_transitions(
            self.agent_params["transitions_per_reward_update"]
        )

        # Update OMPL SimpleSetup object cost function with current learned reward
        self.planner.update_ss_cost(self.reward.cost_fn)
        
        demo_paths = []
        agent_paths = []
        agent_log_probs = []
        for i in range(self.agent_params["transitions_per_reward_update"]):
            # Sample expert transitions (s, a, s')
            # and find optimal path from s' to goal
            ob, ac, log_probs, rewards, next_ob, done = [
                var[i] for var in demo_transitions
            ]
            path, controls = self.planner.plan(next_ob)
            
            path = np.concatenate((ob.reshape(1, self.state_dim), path), axis=0)
            demo_paths.append([path])

            # and find optimal path from s' to goal
            paths = []
            log_probs = []
            for j in range(self.agent_params["agent_actions_per_demo_transition"]):
                # Sample agent transitions (s, a, s') at each expert state s
                agent_ac, _ = self.actor.predict(ob)
                log_prob = utils.get_log_prob(self.actor, agent_ac)
                agent_next_ob = self.env.one_step_transition(ob, agent_ac)
                
                #! "agent_next_ob" is likely not valid  

                # Find optimal path from s' to goal
                path, controls = self.planner.plan(agent_next_ob)
                if path is None:
                    import sys
                    import pickle
                    data = {
                        "agent_ac": agent_ac,
                        "ob": ob,
                        "agent_next_ob": agent_next_ob 
                    }
                    with open(f'./data_{i}{j}.pkl', 'wb') as f:
                        pickle.dump(data, f)
                    print("SAVED!")
                    sys.exit(0)

                path = np.concatenate((ob.reshape(1, self.state_dim), path), axis=0)
                paths.append(path)
                log_probs.append(log_prob)
            agent_paths.append(paths)
            agent_log_probs.append(log_probs)

        demo_paths = self.collate_fn(demo_paths)
        agent_paths = self.collate_fn(agent_paths)
        agent_log_probs = np.array(agent_log_probs)

        reward_logs = []
        for step in range(self.agent_params["reward_updates_per_iter"]):
            reward_logs.append(
                self.reward.update(demo_paths, agent_paths, agent_log_probs)
            )
        return reward_logs

    def collate_fn(self, paths):
        """
        Pad the list of variable-length paths with goal locations
        """
        T = max([len(p) for path_l in paths for p in path_l])
        paths = np.array(
            [
                [np.pad(p, ((0, T - p.shape[0]), (0, 0)), "edge") for p in path_l]
                for path_l in paths
            ]
        )
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
        print("\nTraining agent policy...")
        self.actor.learn(total_timesteps=1000, log_interval=5)
        train_log = {"Policy loss": 0}
        return train_log

    #####################################################
    #####################################################

    def add_to_buffer(self, paths):
        self.demo_buffer.add_rollouts(paths)

    def sample_transitions(self, batch_size: int) -> List[types.Transition]:
        return self.demo_buffer.sample_random_transitions(batch_size)
