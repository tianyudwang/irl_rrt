import gym
import numpy as np 

# Custom env wrapper to change reward function 
class IRLEnv(gym.Wrapper):
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


class PendulumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Maximum value for [theta, theta_dot]
        high = np.array([np.pi, 8], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        
        # Pendulum parameters
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.05
        self.max_angular_velocity = 8.0
        self.max_torque = 2.0

    def step(self, action):
        # Modify obs
        next_state, reward, done, info = self.env.step(action)        
        next_state = self.convert_obs(next_state)
        return next_state, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.convert_obs(obs)
        return obs

    def convert_obs(self, state):
        """
        Convert observation from [np.cos(theta), np.sin(theta), thetadot]
        to [theta, thetadot]
        """
        sin_th, cos_th = state[0], state[1]
        th = np.arctan2(sin_th, cos_th)
        return np.array([th, state[2]])

    def one_step_transition(self, state, action):
        """
        Compute a one-step transition of next state, given current state and action
        """

        th, th_dot, u = state[0], state[1], action
        assert -np.pi <= th <= np.pi, f"State theta is out of bounds: {th}"
        assert -8.0 <= th_dot <= 8.0, f"State theta_dot is out of bounds: {th_dot}"
        assert -2.0 <= u <= 2.0, f"Control input u is out of bounds: {u}"

        newthdot = th_dot + (3.0 * self.g / (2.0 * self.l) * np.sin(th) 
                             + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = np.clip(newthdot, -8.0, 8.0)

        newth = th + newthdot * self.dt
        newth = self.angle_normalize(newth)

        return np.array([newth, newthdot], dtype=np.float32)

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi