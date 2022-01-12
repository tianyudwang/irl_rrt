import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import gym
import d4rl

from AntWrapper import AntMazeFixedStartWrapper, AntMazeFixedGoalWrapper, AntMazeFixStartAndGoalWrapper

from icecream import ic


if __name__ == '__main__':
    
    env = gym.make("antmaze-umaze-v1")  #! v2 has the double goal problem
    # env = AntMazeFixedStartWrapper(env)
    # env = AntMazeFixedGoalWrapper(env)
    env = AntMazeFixStartAndGoalWrapper(env)

    obs = env.reset()
    ic(obs)
    ic(env.unwrapped._goal)
    for i in range(100):
        env.render()
        env.step(env.action_space.sample())
        
        if i % 10 == 0:
            obs = env.reset()
            ic(obs)
            ic(env.unwrapped._goal)
