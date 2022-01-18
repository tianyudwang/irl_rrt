import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import gym
import d4rl

from irl.scripts.wrapper.antWrapper import (
    AntMazeFixedStartWrapper,
    AntMazeFixedGoalWrapper,
    AntMazeFixStartAndGoalWrapper
)

from icecream import ic


if __name__ == '__main__':
    
    env = gym.make("antmaze-umaze-v1")  #! v2 has the double goal problem
    # env = AntMazeFixedStartWrapper(env)
    # env = AntMazeFixedGoalWrapper(env)
    env = AntMazeFixStartAndGoalWrapper(env)

    ic(env.spec.max_episode_steps)      # 700
    obs = env.reset()
    ic(obs)
    ic(env.unwrapped._goal)
    for i in range(1000):
        env.render()
        env.step(env.action_space.sample())
        
        if i % 10 == 0:
            obs = env.reset()
            assert (obs[0], obs[1]) == (0, 0)
            ic(obs)
            ic(env.unwrapped._goal)