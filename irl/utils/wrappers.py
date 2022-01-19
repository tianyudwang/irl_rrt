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


# class PendulumWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         # Maximum value for [theta, theta_dot]
#         high = np.array([np.pi, 8], dtype=np.float32)
#         self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        
#         # Pendulum parameters
#         self.g = 10.0
#         self.m = 1.0
#         self.l = 1.0
#         self.dt = 0.05
#         self.max_angular_velocity = 8.0
#         self.max_torque = 2.0

#     def step(self, action):
#         # Modify obs
#         next_state, reward, done, info = self.env.step(action)        
#         next_state = self.convert_obs(next_state)
#         return next_state, reward, done, info

#     def reset(self):
#         obs = self.env.reset()
#         obs = self.convert_obs(obs)
#         return obs

#     def convert_obs(self, state):
#         """
#         Convert observation from [np.cos(theta), np.sin(theta), thetadot]
#         to [theta, thetadot]
#         """
#         sin_th, cos_th = state[0], state[1]
#         th = np.arctan2(sin_th, cos_th)
#         return np.array([th, state[2]])

#     def one_step_transition(self, state, action):
#         """
#         Compute a one-step transition of next state, given current state and action
#         """

#         th, th_dot, u = state[0], state[1], action
#         assert -np.pi <= th <= np.pi, f"State theta is out of bounds: {th}"
#         assert -8.0 <= th_dot <= 8.0, f"State theta_dot is out of bounds: {th_dot}"
#         assert -2.0 <= u <= 2.0, f"Control input u is out of bounds: {u}"

#         newthdot = th_dot + (3.0 * self.g / (2.0 * self.l) * np.sin(th) 
#                              + 3.0 / (self.m * self.l ** 2) * u) * self.dt
#         newthdot = np.clip(newthdot, -8.0, 8.0)

#         newth = th + newthdot * self.dt
#         newth = self.angle_normalize(newth)

#         return np.array([newth, newthdot], dtype=np.float32)

#     def angle_normalize(self, x):
#         return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumWrapper(gym.Wrapper):
    """
    Wrapper for Pendulum-v1
    1. Change observation from [cos(theta), sin(theta), theta_dot] to [theta, theta_dot]
    2. Change time-limit to first exit
    3. Expose a one_step_transition function which computes next state 
        given current state and action
    """
    def __init__(self, env):
        super().__init__(env)

        self.high = np.array([np.pi, 8], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-self.high, 
            high=self.high, 
            dtype=np.float32
        )

    def is_goal(self, state):
        th, thdot = state

        if np.abs(th) <= 0.1 and np.abs(thdot) <= 0.5:
            return True 
        else:
            return False

    def step(self, action):
        obs, rew, done, info = super().step(action)

        # In original implementation, thdot is clipped but th is not
        th, thdot = self.unwrapped.state
        th = self.angle_normalize(th)
        obs = np.array([th, thdot], dtype=np.float32)

        # Check if pendulum is inverted
        if self.is_goal(obs):
            done = True

        return obs, rew, done, info

    def reset(self, state=None):
        super().reset()
        if state is not None:
            high = np.array([np.pi, 8])
            state = np.clip(state, -high, high)
            self.unwrapped.state = state

        # print(f"Reset location at {self.unwrapped.state}")
        return self.unwrapped.state

    def one_step_transition(self, state, action):

        th, thdot = np.clip(state, -self.high, self.high)
        u = np.clip(action, -self.unwrapped.max_torque, self.unwrapped.max_torque)[0]

        g = self.unwrapped.g
        m = self.unwrapped.m
        l = self.unwrapped.l
        dt = self.unwrapped.dt

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th)+ 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.unwrapped.max_speed, self.unwrapped.max_speed)

        newth = self.angle_normalize(th + newthdot * dt)
        next_state = np.array([newth, newthdot])
        return next_state

    def angle_normalize(self, x):
        """Normalize angle between -pi and pi"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi
