
import numpy as np
import gym
from stable_baselines3 import SAC

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from irl.agents.base_agent import BaseAgent 
from irl.scripts.replay_buffer import ReplayBuffer
from irl.agents.mlp_reward import MLPReward
import irl.scripts.pytorch_util as ptu 
import irl.scripts.utils as utils

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
        #reward = self.reward(self.last_obs, obs).item()
        self.last_obs = obs 
        return obs, reward, done, info

    def reset(self):
        self.last_obs = self.env.reset()
        return self.last_obs

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
        self.init_RRT()

        # Replay buffer to hold demo transitions (maximum transitions)
        self.demo_buffer = ReplayBuffer(1000)
        self.sample_buffer = ReplayBuffer(1000)

    def train_reward(self):
        """
        Train the reward function
        """
        print('\nTraining agent reward function...')
        num_transitions = self.agent_params['transitions_per_reward_update']
        demo_transitions = self.sample_transitions(num_transitions, demo=True)
        agent_transitions = self.sample_transitions(num_transitions)

        # Update OMPL SimpleSetup object cost function with current learned reward
        self.update_ss_cost(self.reward.cost_fn)

        demo_paths = self.plan_optimal_paths(demo_transitions)
        agent_paths = self.plan_optimal_paths(agent_transitions)

        reward_logs = []
        for i in range(self.agent_params['reward_updates_per_iter']):
            reward_logs.append(self.reward.update(demo_paths, agent_paths))
        return reward_logs

    def plan_optimal_paths(self, transitions):
        """
        For each transition (s, a, s'), we find the optimal path from s' to goal
        """
        num_transitions = transitions[0].shape[0]
        paths = []
        for i in range(num_transitions):
            obs, ac, rew, next_obs, done = [var[i] for var in transitions]
            path = self.RRT_plan(next_obs)
            path = np.concatenate((obs.reshape(1, self.state_dim), path), axis=0)
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

        # Simple setup instance that contains the 
        ss = og.SimpleSetup(si)

        # Set the agent goal state
        goal = ob.State(space)
        goal[0], goal[1] = 1.0, 1.0  
        ss.setGoalState(goal)

        # Set up RRT* planner
        ss.setPlanner(og.RRTstar(si))
        self.ss = ss

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)

    def RRT_plan(self, start_state, solveTime=0.1):
        """
        :param ss: OMPL SimpleSetup object, initialized with RRT* planner
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
        solved = self.ss.solve(solveTime)
        if solved:
            path = self.ss.getSolutionPath().printAsMatrix() 
            path = np.fromstring(path, dtype=float, sep='\n').reshape(-1, self.state_dim)
        else:
            print("OMPL is not able to solve under current cost function")
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

