import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from matplotlib import pyplot as plt

class NavEnv(gym.Env):
    """
    2D continuous box environment for navigation
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, size=1, seed=1337):

        # the box size is (2*size, 2*size)
        self.size = size
        self.max_steps = 100
        self.pos_dim = 2
        self.act_dim = 2
        self.obs_num = 64
        self.dt = 0.1

        # Observations are the position
        self.observation_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(self.pos_dim,),
            dtype="float32"
        )

        # Actions are 2D displacement
        self.action_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(self.act_dim,),
            dtype="float32"
        )

        # Window to use for human rendering mode
        self.window = None

        # Mission of the task
        self.mission = 'Avoid the origin and navigate to top right'
        self.goal = np.array([self.size, self.size], dtype=float)

        # Initialize seed
        self.seed(seed=seed)

        # Initialize reward map
        self.init_reward()

        # Initialize the state
        self.reset()

    def step(self, action):
        """
        One step transition 
        """

        self.step_count += 1
        self.update_states(action)
        #self.pos = self.one_step_transition(self.pos, action)

        # Observation is the position 
        obs = np.copy(self.pos)
        reward = self.eval_reward(obs)

        reward *= self.gamma**self.step_count 

        done = False
        # Done if reached maximum steps or reached goal 
        if self.step_count >= self.max_steps or self.is_goal(self.pos):
            done = True

        return obs, reward, done, {}

    def preprocess_obs(self, agent_pos):
        """
        Add the agent position to the reward map
        """
        obs = self.reward_map.copy()
        agent_idx = ((agent_pos + self.size) / (2 * self.size) * (self.obs_num-1)).astype(int)

        # Set the agent to be red
        obs[agent_idx[0], agent_idx[1]] = np.array([255, 0, 0])
        return obs

    def one_step_transition(self, pos, action):
        """
        Update the position with a simple dynamics model
        x = x + u * dt
        Force the position and action within bounds [-self.size, self.size]
        """
        action[action > self.size] = self.size
        action[action < -self.size] = -self.size
        pos += action * self.dt 
        pos[pos > self.size] = self.size
        pos[pos < -self.size] = -self.size
        return pos

    def update_states(self, action):
        """
        Update the position with a simple dynamics model
        x = x + u * dt
        Force the position and action within bounds [-self.size, self.size]
        """
        action[action > self.size] = self.size
        action[action < -self.size] = -self.size
        self.pos += action * self.dt 
        self.pos[self.pos > self.size] = self.size
        self.pos[self.pos < -self.size] = -self.size

    def reset(self):
        """
        Randomly spawn a starting location / Spawn at lower left
        """
#        self.pos = np.random.uniform(
#            low=-self.size, 
#            high=self.size, 
#            size=self.pos_dim
#        )
        self.pos = np.array([-self.size, -self.size], dtype=float)

        # Step count since episode start
        self.step_count = 0

        # Observation is the position
        obs = self.pos

        return obs

    def render(self, mode='human'):
        """
        Visualize the trajectory on the reward map
        """

        if mode == 'human' and not self.window:
            import gym_nav.window
            self.window = gym_nav.window.Window('gym_nav')
            self.window.show(block=False)

        img = self.preprocess_obs(self.pos)

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        elif mode == 'rgb_array':
            return img 


    def close(self):
        if self.window:
            self.window.close()
        return 

    def is_goal(self, pos, eps=0.1):
        if np.linalg.norm(pos - self.goal) <= eps:
            return True
        else:
            return False 

    def init_reward(self):
        """
        The reward is a Gaussian at [0, 1]
        
        """
        # discount
        self.gamma = 0.99
        
        self.mixtures = {
            'A': [-2],
            'mu': [np.array([-self.size, self.size], dtype=float)],
            'std': [self.size]
        }

        # Increase obs_num for higher resolution
        num = self.obs_num
        x = np.linspace(-self.size, self.size, num=num)
        y = np.linspace(-self.size, self.size, num=num)

        # TODO: vectorize this computation
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                Z[i,j] = self.eval_gaussian(np.array([X[i,j], Y[i,j]]))
        # rescale values to [0, 255]
        self.reward_min, self.reward_max = np.min(Z), np.max(Z)
        Z = ((Z - self.reward_min) / (self.reward_max - self.reward_min) * 255).astype(np.uint8)
        # convert grayscale to rgb by stacking three channels
        self.reward_map = np.stack((Z, Z, Z), axis=-1)

    def eval_gaussian(self, x):
        """
        Evaluate the value of mixture of gaussian functions at location x
        """ 
        ans = 0
        for mu, std, A in zip(self.mixtures['mu'], self.mixtures['std'], self.mixtures['A']):
            ans += A * np.exp(-(x - mu).T @ (x - mu) / (2 * std**2))
        return ans

    def eval_reward(self, x):
        """
        Evaluate the reward at the current position x
        """
        reward = self.eval_gaussian(x)

        # add +5 reward if reached goal
        if self.is_goal(x):
            reward += 10

#        # shape reward to [-1, 1] to assist learning
#        reward = (reward - self.reward_min) / (self.reward_max - self.reward_min) * 2 - 1
        return reward