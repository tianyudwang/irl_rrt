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

class Maze2DFirstExitWrapper(gym.Wrapper):
    """Set done variable to True if reached goal target"""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        ob, reward, done, info = super().step(action)
        if done or np.linalg.norm(ob[0:2] - self.unwrapped._target) <= 0.5:
            done = True
        else:
            done = False
        return ob, reward, done, info


class Maze2DTransitionWrapper(gym.Wrapper):
    """Expose an one-step transition function"""
    def __init__(self, env):
        super().__init__(env)
        
        self.state_low = np.array([0.5, 0.5, -5., -5.])
        self.state_high = np.array([3.5, 3.5, 5., 5.])

        size = 0.1
        # Square extents
        self.square_x_min = 0.5 + size
        self.square_x_max = 3.5 - size
        self.square_y_min = 0.5 + size
        self.square_y_max = 3.5 - size   

        # Rectangle extents
        self.rect_x_min = 1.5 - size
        self.rect_x_max = 2.5 + size
        self.rect_y_min = 0.5 + size
        self.rect_y_max = 2.5 + size

    def is_valid(self, state):
        in_square = ((self.square_x_min <= state[0] <= self.square_x_max) 
            and (self.square_y_min <= state[1] <= self.square_y_max))
        assert in_square, (
            f"State {state} not in square"
        )

        in_rect = ((self.rect_x_min <= state[0] <= self.rect_x_max) 
            and (self.rect_y_min <= state[1] <= self.rect_y_max))
        assert not in_rect, (
            f"State {state} in rectangle"
        )
        return True

    def one_step_transition(self, state, action):
        """Set mujoco simulator to state and apply action to get next state"""
        qpos, qvel = state[:2], state[2:]
        self.unwrapped.set_state(qpos, qvel)
        obs, _, _, _ = self.unwrapped.step(action)
        obs = np.clip(obs, self.state_low, self.state_high)
        assert self.is_valid(obs), "Next state not valid after one_step_transition"
        return obs
