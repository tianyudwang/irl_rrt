import gym
import numpy as np

from icecream import ic


if __name__ == "__main__":
    #
    env1 = gym.make("Ant-v3")
    env2 = gym.make("Hopper-v3")
    env3 = gym.make("HalfCheetah-v2")

    # ic(env1)
    # ic(env1.observation_space)
    # ic(env1.action_space)

    ic(env2)
    ic(env2.observation_space)
    ic(env2.action_space)

    # ic(env3)
    # ic(env3.observation_space)
    # ic(env3.action_space)

    obs_ = env2.reset()
    act_ = env2.action_space.sample()
    # posbefore = self.sim.data.qpos[0]
    # posafter, height, ang = self.sim.data.qpos[0:3]
    ic(obs_)
    ic(act_)
