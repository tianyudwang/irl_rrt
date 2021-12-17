import pickle
import time

import gym
import mujoco_maze
import numpy as np
from icecream import ic

from irl.scripts.wrapper.remove_timeDim_wrapper import RemovTimeFeatureWrapper
from irl.scripts.wrapper.one_step_PointUMaze_wrapper import PointUMazeOneStepTransitionWrapper




if __name__ == "__main__":

    env = gym.make("PointUMaze-v0")
    # env = PointUMazeOneStepTransitionWrapper(RemovTimeFeatureWrapper(maze_env))

    obs = env.reset()
    with open("./data.pkl", "rb") as f:
        data = pickle.load(f)

    # ic(env.isValid(data["ob"]))
    # ic(env.satisfiedStateBounds(data["ob"]))
    # ic(env.satisfiedControlBounds(data["agent_ac"]))

    # next_obs = env.one_step_transition(state=data["ob"], action= data["agent_ac"])
    # ic(next_obs)
    # ic(env.isValid(next_obs))

    nq = env.unwrapped.wrapped_env.sim.model.nq
    nv = env.unwrapped.wrapped_env.sim.model.nv
    # qpos_temp = data["ob"][: nq].copy()
    # qvel_temp = data["ob"][nq :nq + nv].copy()
    # env.unwrapped.wrapped_env.set_state(qpos=qpos_temp, qvel=qvel_temp)

    
    qpos_temp = data["agent_next_ob"][: nq].copy()
    qvel_temp = data["agent_next_ob"][nq :nq + nv].copy()
    env.unwrapped.wrapped_env.set_state(qpos=qpos_temp, qvel=qvel_temp)
    env.render()
    time.sleep(10)
    ic(env.unwrapped.wrapped_env.RADIUS)
    