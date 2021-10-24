import sys
import os
import time
import pathlib
from typing import Any, Dict

import yaml

# from PIL import Image
# import imageio

import gym
import gym.envs.robotics.utils as robotics_utils

import numpy as np
import matplotlib.pyplot as plt

from irl.scripts import ompl_utils
from irl.mujoco_ompl_py.mujoco_ompl_interface import *


try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(
        0,
        join(dirname(dirname(dirname(dirname(abspath(__file__))))), "ompl/py-bindings"),
    )
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og


def visulallzie_env(env: gym.Env, joint_idx: int) -> None:
    """
    Visualize the environment
    :param env:
    :param render:
    :param render_video:
    :return:
    """
    while True:
        try:
            env.render()
            env.sim.data.qpos[joint_idx] += 0.1
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":

    args = ompl_utils.CLI()

    ic(args.planner)

    # Current directory
    path = pathlib.Path(__file__).parent.resolve()

    # Set the OMPL log level
    ompl_utils.setLogLevel(args.info)

    # Raise overflow / underflow warnings to errors
    np.seterr(all="raise")

    # Set the random seed (especially ou.RNG)
    ompl_utils.setRandomSeed(args.seed)

    # Create the environment
    env = gym.make(args.env_id)
    env.seed(args.seed)

    # Initialize the environment
    env.reset()

    # Obtain the start state in Joint Space
    q_pos_start = env.sim.get_state().qpos
    q_vel_start = env.sim.get_state().qvel

    # Load the goal state in yaml file
    with open(path / "inverted_pendulum_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    
    # Set the parameters of planning
    param = dict(
        start=np.concatenate([q_pos_start, q_vel_start]),
        goal=config["goal"],
        include_velocity = True,
        plannerType=args.planner,
        state_dim=env.observation_space.shape[0],
    )

    ic(param["start"])
    ic(param["goal"])
    
    # Setup
    ss = ompl_utils.init_planning(env, param)

    si = ss.getSpaceInformation()
    space = si.getStateSpace()
    joints = getJointInfo(env.sim.model)


    controlPath, _, _ = ompl_utils.plan(ss, param, args.runtime)


    # # Get controls
    controls = controlPath.getControls()
    
    control_count = controlPath.getControlCount()
    ic(control_count)
    U = [u[0] for u in controls]
    ic(U)
    for u in U:
        obs, rew, done, info = env.step(u)

        if args.render or args.render_video:
            try:
                if args.render_video:
                    img_array = env.render(mode="rgb_array")
                else:
                    env.render(mode="human")
                    time.sleep(1)
            except KeyboardInterrupt:
                break