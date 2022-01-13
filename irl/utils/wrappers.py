import gym
import numpy as np 

class IRLWrapper(gym.RewardWrapper):
    """Custom gym environment wrapper to change reward function"""
    def __init__(self, env, reward):
        super().__init__(env)
        assert callable(reward)
        self.reward = reward

    def step(self, action):
        """Override the true environment reward with learned reward"""
        obs, reward, done, info = self.env.step(action)
        reward = self.reward.reward_fn(self.last_obs, obs)
        self.last_obs = obs.copy()
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs.copy()
        return obs

class MazeIRLWrapper(IRLWrapper):
    """Wrapper to provide one step transition function"""
    def __init__(self, env, reward):
        super().__init__(env, reward)

    def one_step_transition(self, state, action):
        """Compute one step transition explicitly given current state and action"""
        raise NotImplementedError

class Maze2DFixedStartWrapper(gym.Wrapper):
    """Fix the start location at (3, 1) in maze2d-umaze-v1"""
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped.reset_model = self.reset_model

    def reset_model(self):
        reset_location = np.array([3, 1]).astype(self.observation_space.dtype)
        qpos = reset_location #+ self.unwrapped.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel #+ self.unwrapped.np_random.randn(self.model.nv) * .1
        self.unwrapped.set_state(qpos, qvel)
        if self.reset_target:
            self.unwrapped.set_target()
        return self.unwrapped._get_obs()

class Maze2DFixedLocationWrapper(gym.Wrapper):
    """Does not include noise in reset_to_location function in MazeEnv"""
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped.reset_to_location = self.reset_to_location

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location 
        qvel = self.init_qvel 
        self.unwrapped.set_state(qpos, qvel)
        return self.unwrapped._get_obs()

class AntMazeFixedGoalWrapper(gym.Wrapper):
    """Fix the goal location at (0, 8) for antmaze-umaze-v1"""
    def __init__(self, env): 
        # self.unwrapped.set_target_goal = self.set_target_goal()
        pass

    def set_target_goal(self, goal_input=None):

        self.unwrapped.target_goal = (0, 8)
      
        print ('Target Goal: ', self.unwrapped.target_goal)
        ## Make sure that the goal used in self._goal is also reset:
        self.unwrapped._goal = self.unwrapped.target_goal