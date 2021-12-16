import pickle

import gym
import mujoco_maze
import numpy as np
from icecream import ic

from irl.scripts.wrapper.remove_timeDim_wrapper import RemovTimeFeatureWrapper
from irl.scripts.wrapper.one_step_PointUMaze_wrapper import PointUMazeOneStepTransitionWrapper




if __name__ == "__main__":

    env = gym.make("PointUMaze-v0")
    env = PointUMazeOneStepTransitionWrapper(RemovTimeFeatureWrapper(env))

    with open("./data.pkl", "rb") as f:
        data = pickle.load(f)
    ic(data)

    ic(env.isValid(data["ob"]))
    ic(env.satisfiedStateBounds(data["ob"]))
    ic(env.satisfiedControlBounds(data["agent_ac"]))