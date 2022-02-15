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


class ReacherWrapper(gym.Wrapper):
    """
    Wrapper for Reacher-v2, https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py
    Timelimit with max_episode_steps = 50
    Observation changed to the following:
    | Num | Observation                             | Min        | Max       | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------------------------|------------|-----------|-----------|-------|--------------------|
    | 0   | angle of the first arm                  | -Inf       | Inf       | joint0    | hinge | unitless |
    | 1   | angle of the second arm                 | -Inf       | Inf       | joint1    | hinge | unitless |
    | 2   | angular velocity of the first arm       | -Inf       | Inf       | joint0    | hinge | angular velocity (rad/s) |
    | 3   | angular velocity of the second arm      | -Inf       | Inf       | joint1    | hinge | angular velocity (rad/s) |
    | 4   | x-value of position_fingertip           | -Inf       | Inf       | NA        | slide | position (m) |
    | 5   | y-value of position_fingertip           | -Inf       | Inf       | NA        | slide | position (m) |
    | 6   | x-coorddinate of the target             | -Inf       | Inf       | target_x  | slide | position (m) |
    | 7   | y-coorddinate of the target             | -Inf       | Inf       | target_y  | slide | position (m) |

    """

    def __init__(self, env):
        super().__init__(env)

        self.high = np.array([
            np.pi,
            np.pi,
            np.inf,
            np.inf,
            0.21,
            0.21,
            0.21,
            0.21
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-self.high,
            high=self.high,
            dtype=np.float32
        )

    def is_goal(self, state):
        raise NotImplementedError

    def angle_normalize(self, x):
        """Normalize angle between -pi and pi"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def step(self, action):
        ob, rew, done, info = super().step(action)
        ob = self._get_obs()
        info = {
            'qpos': self.unwrapped.sim.data.qpos.flat[:].copy(),
            'qvel': self.unwrapped.sim.data.qvel.flat[:].copy()
        }
        return ob, rew, done, info

    def reset(self):
        super().reset()
        return self._get_obs()

    def _get_obs(self):
        theta = self.angle_normalize(self.unwrapped.sim.data.qpos.flat[:2])
        ob = np.concatenate([
            theta,
            self.unwrapped.sim.data.qvel.flat[:2],
            self.unwrapped.get_body_com("fingertip")[:2],
            self.unwrapped.sim.data.qpos.flat[2:],
        ])
        return ob.astype(np.float32).copy()

    def one_step_transition(self, state, action):
        """Set mujoco simulator to state and apply action to get next state"""
        qpos, qvel = np.zeros(4), np.zeros(4)
        qpos[:2] = self.angle_normalize(state[:2])
        qpos[2:4] = state[-2:]
        qvel[:2] = state[2:4]

        self.unwrapped.set_state(qpos, qvel)
        ob, rew, done, info = self.step(action)
        return ob
