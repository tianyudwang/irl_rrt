import gym
import numpy as np 

class RemovTimeFeatureWrapper(gym.ObservationWrapper):
    """
    Remove the the last dimension of environment
    Note this wrapper will not check if the last dim is time.
    It will just remove it. 
    :param env: Gym env to wrap.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        low, high = obs_space.low[:-1], obs_space.high[:-1]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
    
    def observation(self, obs):
        # Remove the last dimension
        return obs[:-1]
    
        
    
    