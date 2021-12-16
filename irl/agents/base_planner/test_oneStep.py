import pickle
import time

import gym
import mujoco_maze
import numpy as np
from icecream import ic

from irl.scripts.wrapper.remove_timeDim_wrapper import RemovTimeFeatureWrapper
from irl.scripts.wrapper.one_step_PointUMaze_wrapper import PointUMazeOneStepTransitionWrapper




if __name__ == "__main__":

    maze_env = gym.make("PointUMaze-v0")
    env = PointUMazeOneStepTransitionWrapper(RemovTimeFeatureWrapper(maze_env))

    obs = env.reset()
    env.render()
    with open("./data.pkl", "rb") as f:
        data = pickle.load(f)
    ic(data)

    ic(env.isValid(data["ob"]))
    ic(env.satisfiedStateBounds(data["ob"]))
    ic(env.satisfiedControlBounds(data["agent_ac"]))

    next_obs = env.one_step_transition(state=data["ob"], action= data["agent_ac"])
    ic(next_obs)
    ic(env.isValid(next_obs))


    inner_obs = env.dummy_step(state=data["ob"], action=data["agent_ac"])
    ic(inner_obs)
    ic(env.isValid(inner_obs))
    time.sleep(10)