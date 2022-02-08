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
        nn_reward = self.reward.reward_fn(self.last_obs, obs)
        self.last_obs = obs.copy()
        # print(reward, nn_reward)
        return obs, nn_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs.copy()
        return obs


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
        if np.linalg.norm(state) <= 0.1:
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


class ReacherWrapper(gym.Wrapper):
    """
    Wrapper for Reacher-v2, https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py
    Timelimit with max_episode_steps = 50
    Observation changed to the following:
    | Num | Observation                             | Min        | Max       | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------------------------|------------|-----------|-----------|-------|--------------------|
    | 0   | angle of the first arm                  | -Inf       | Inf       | joint0    | hinge | unitless |
    | 1   | angle of the second arm                 | -Inf       | Inf       | joint1    | hinge | unitless |
    | 2   | x-coorddinate of the target             | -Inf       | Inf       | target_x  | slide | position (m) |
    | 3   | y-coorddinate of the target             | -Inf       | Inf       | target_y  | slide | position (m) |
    | 4   | angular velocity of the first arm       | -Inf       | Inf       | joint0    | hinge | angular velocity (rad/s) |
    | 5   | angular velocity of the second arm      | -Inf       | Inf       | joint1    | hinge | angular velocity (rad/s) |
    | 6   | x-value of position_fingertip           | -Inf       | Inf       | NA        | slide | position (m) |
    | 7   | y-value of position_fingertip           | -Inf       | Inf       | NA        | slide | position (m) |

    """

    def __init__(self, env):
        super().__init__(env)

        self.high = np.array([
            np.pi,
            np.pi,
            0.2,
            0.2,
            np.inf,
            np.inf,
            0.2,
            0.2
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-self.high,
            high=self.high,
            dtype=np.float32
        )

    def is_goal(self, state):
        raise NotImplementedError

    def step(self, action):
        ob, rew, done, info = super().step(action)
        ob = self._get_obs()
        return ob, rew, done, info

    def reset(self):
        super().reset()
        return self._get_obs()

    def _get_obs(self):
        theta = self.unwrapped.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                theta,
                self.unwrapped.sim.data.qpos.flat[2:],
                self.unwrapped.sim.data.qvel.flat[:2],
                self.unwrapped.get_body_com("fingertip")[:2],
            ]
        )