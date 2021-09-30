
import numpy as np
import gym
from stable_baselines3 import SAC
import multiprocessing as mp

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from irl.agents.base_agent import BaseAgent 
from irl.scripts.replay_buffer import ReplayBuffer
from irl.agents.mlp_reward import MLPReward
import irl.scripts.pytorch_util as ptu 
import irl.scripts.utils as utils

import time 

# Custom env wrapper to change reward function
class NavIRLEnv(gym.Wrapper):
    def __init__(self, env, reward):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.reward = reward

    def step(self, action):
        """
        Override the true environment reward with learned reward
        """
        obs, reward, done, info = self.env.step(action)
        reward = self.reward.reward_fn(self.last_obs, obs)
        self.last_obs = obs.copy()
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs.copy()
        return obs

## @cond IGNORE
# Our "collision checker". For this demo, our robot's state space
# lies in [-1,1]x[-1,1], without any obstacles. The checker trivially
# returns true for any state
class ValidityChecker(ob.StateValidityChecker):
    # Returns whether the given state's position overlaps the
    # circular obstacle
    def isValid(self, state):
        return True

## Defines an optimization objective by computing the cost of motion between 
# two endpoints.
class IRLCostObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super(IRLCostObjective, self).__init__(si)
        self.cost_fn = cost_fn
    
    def motionCost(self, s1, s2):
        s1 = np.array([s1[0], s1[1]])
        s2 = np.array([s2[0], s2[1]])
        c = self.cost_fn(s1, s2)
        return ob.Cost(c)

def getIRLCostObjective(si, cost_fn):
    return IRLCostObjective(si, cost_fn)

class IRL_Agent(BaseAgent):
    def __init__(self, env, agent_params):
        super(IRL_Agent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # reward function
        self.reward = MLPReward(
            self.agent_params['ob_dim'],
            self.agent_params['ac_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['output_size'],
            self.agent_params['learning_rate']
        )
        
        # create a wrapper env with learned reward
        self.irl_env = NavIRLEnv(self.env, self.reward)

        # actor/policy with wrapped env
        self.actor = SAC("MlpPolicy", self.irl_env, verbose=1)

        # RRT
        self.state_dim = self.agent_params['ob_dim']
        self.goal = np.array([1.0, 1.0])
        self.init_RRT()

        # Replay buffer to hold demo transitions (maximum transitions)
        self.demo_buffer = ReplayBuffer(1000)
        self.sample_buffer = ReplayBuffer(1000)

    def train_reward(self):
        """
        Train the reward function
        """
        print('\nTraining agent reward function...')
        demo_transitions = self.sample_transitions(
            self.agent_params['transitions_per_reward_update'], 
            demo=True)

        agent_transitions = self.sample_transitions(
            self.agent_params['transitions_per_reward_update'])

        # Update OMPL SimpleSetup object cost function with current learned reward
        self.update_ss_cost(self.reward.cost_fn)

        demo_paths = []
        agent_paths = []
        agent_log_probs = []

        start_time = time.time()
        for i in range(self.agent_params['transitions_per_reward_update']):
            # Sample expert transitions (s, a, s')
            # and find optimal path from s' to goal
            ob, ac, log_probs, rewards, next_ob, done = [var[i] for var in demo_transitions]
            path = self.RRT_plan(next_ob)
            path = np.concatenate((ob.reshape(1, self.state_dim), path), axis=0)
            demo_paths.append([path])
            
            # and find optimal path from s'_a to goal
            paths = []
            log_probs = []
            for j in range(self.agent_params['agent_actions_per_demo_transition']):
                # Sample agent transitions (s, a, s') at each expert state s
                agent_ac, _ = self.actor.predict(ob)
                log_prob = utils.get_log_prob(self.actor, agent_ac)
                agent_next_ob = self.env.one_step_transition(ob, agent_ac)

                # Find optimal path from s' to goal
                path = self.RRT_plan(agent_next_ob)
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

    def plan_optimal_paths(self, transitions):
        """
        For each transition (s, a, s'), we find the optimal path from s' to goal
        """
        num_transitions = transitions[0].shape[0]
        paths = []
        for i in range(num_transitions):
            obs, ac, rew, next_obs, done = [var[i] for var in transitions]
            path = self.RRT_plan(next_obs)
            paths.append(path)
        return paths

    def init_RRT(self):
        """
        Initialize an ompl::geometric::SimpleSetup instance
        Check out https://ompl.kavrakilab.org/genericPlanning.html
        """
        # Set log to warn/info/debug
        ou.setLogLevel(ou.LOG_WARN)
        # Construct the state space in which we're planning. We're
        # planning in [-1,1]x[-1,1], a subset of R^2.
        space = ob.RealVectorStateSpace(self.state_dim)  
        # Set the bounds of space to be in [-1,1].
        space.setBounds(-1.0, 1.0)   
        self.space = space

        # Construct a space information instance for this state space
        si = ob.SpaceInformation(space) 
        # Set the object used to check which states in the space are valid
        validityChecker = ValidityChecker(si)
        si.setStateValidityChecker(validityChecker) 
        si.setup()
        self.si = si

        # Simple setup instance that contains the space information
        ss = og.SimpleSetup(si)

        # Set the agent goal state
        goal = ob.State(space)
        goal[0], goal[1] = self.goal[0], self.goal[1]  
        ss.setGoalState(goal)

        # Set up RRT* planner
        planner = og.RRTstar(si)
        # Set the maximum length of a motion
        planner.setRange(0.1)
        ss.setPlanner(planner)
        self.ss = ss

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)

    def RRT_plan(self, start_state, solveTime=0.5):
        """
        :param start_state: start location of the planning problem
        :param solveTime: allowed planning budget
        :return:
            path
        """
        # Clear previous planning data, does not affect settings and start/goal
        self.ss.clear()

        # Reset the start state
        start = ob.State(self.space)
        start[0], start[1] = start_state[0].item(), start_state[1].item() 
        self.ss.setStartState(start)

        # solve and get optimal path
        # TODO: current termination condition is a fixed amount of time for planning
        # Change to exactSolnPlannerTerminationCondition when an exact but suboptimal 
        # path is found
        while not self.ss.getProblemDefinition().hasExactSolution():
            solved = self.ss.solve(solveTime)

        if solved:
            path = self.ss.getSolutionPath().printAsMatrix() 
            path = np.fromstring(path, dtype=float, sep='\n').reshape(-1, self.state_dim)
        else:
            raise ValueError("OMPL is not able to solve under current cost function")
            path = None
        return path


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
    
    def add_to_buffer(self, paths, demo=False):
        """
        Add paths to demo buffer
        """
        if demo:
            self.demo_buffer.add_rollouts(paths)
        else:
            self.sample_buffer.add_rollouts(paths)

    def sample_rollouts(self, batch_size, demo=False):
        """
        Sample paths from demo buffer
        """
        if demo:
            return self.demo_buffer.sample_recent_rollouts(batch_size)
        else:
            return self.sample_buffer.sample_recent_rollouts(batch_size)

    def sample_transitions(self, batch_size, demo=False):
        """
        Sample transitions from demo buffer
        returns observations, actions, rewards, next_observations, terminals
        """
        if demo:
            return self.demo_buffer.sample_random_data(batch_size)
        else:
            return self.sample_buffer.sample_recent_data(batch_size)

