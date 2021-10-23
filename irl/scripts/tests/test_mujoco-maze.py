import time

import gym
import mujoco_maze



if __name__ == '__main__':
    env = gym.make("PointUMaze-v0")
    env.reset()
    while True:
        env.render()
    