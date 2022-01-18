import gym
import numpy as np 

class AntMazeFixedStartWrapper(gym.Wrapper):
    """Fix the start location at (0, 0) in antmaze-umaze-v1"""
    def __init__(self, env, start):
        super().__init__(env)
        self.unwrapped.wrapped_env.reset_model = self.reset_model
        
        self._non_zero_reset = self.unwrapped.wrapped_env._non_zero_reset

    def reset_model(self):
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
    
        if self._non_zero_reset:
            """Now the reset is supposed to be to a non-zero location"""
            reset_location = self.unwrapped.wrapped_env._get_reset_location()
            qpos[:2] = reset_location
        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.unwrapped.wrapped_env.set_state(qpos, qvel)
        
        return self.unwrapped.wrapped_env._get_obs()    
    
    
class AntMazeFixedGoalWrapper(gym.Wrapper):
    def __init__(self, env, goal_pos = (0, 8)):
        super().__init__(env)
        self.goal_pos = goal_pos
        self.set_target(self.goal_pos)

    def reset(self):
        self.set_target(self.goal_pos)
        return super().reset()


class AntMazeFixStartAndGoalWrapper(AntMazeFixedStartWrapper, AntMazeFixedGoalWrapper):
    def __init__(self, env, goal_pos = (0, 8)):
        super().__init__(env, goal_pos)