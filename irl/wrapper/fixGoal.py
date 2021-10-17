import gym
import numpy as np


class FixGoal(gym.Wrapper):
    def __init__(self, env, pos=(1.3040752, 0.74440193, 0.66095406)):
        super().__init__(env)
        self.env = env
        assert len(pos) == 3, "pos should be a list of 3 elements representing x, y, z positions"
        if not isinstance(pos, np.ndarray):
            pos = np.asarray(pos, dtype=np.float32)
        self.pos = pos

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        achieved_goal = obs[3:6]
        reward = self.compute_reward(achieved_goal, self.env.goal)
        return obs, reward, done, info

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert isinstance(goal_a, np.ndarray) and isinstance(goal_b, np.ndarray)
        assert goal_a.shape == goal_b.shape
        assert len(goal_a) == len(goal_b) == 3
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info=None):
        d = self.goal_distance(achieved_goal, goal)
        if self.env.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def reset(self):
        # Verified obs is slightly changeing but unable to distingush from render 
        obs = self.env.reset()
        self.env.goal[0] = self.pos[0]
        self.env.goal[1] = self.pos[1]
        self.env.goal[2] = self.pos[2]

        obs[0:3] = self.env.goal.copy()
        return obs