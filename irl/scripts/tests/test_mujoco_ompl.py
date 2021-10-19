import os
import math
import time
import pathlib

import gym
import numpy as np
import mujoco_py

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
        0, join(dirname(dirname(dirname(abspath(__file__)))), "ompl", "py-bindings")
    )
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noq

from irl.mujoco_ompl_py.mujoco_wrapper import *
from irl.mujoco_ompl_py.mujoco_ompl_interface import *


MODEL_XML_PATH = os.path.join("fetch", "reach.xml")

N_SUBSTEPS = 20
"""
class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
"""

if __name__ == "__main__":
    env = gym.make("FetchReach-v1")

    # current_dir
    path = pathlib.Path(__file__).parent.resolve()

    # assets dir (model files)
    model_dir = path.parent / "assets"

    model_fullpath = os.path.join(model_dir, MODEL_XML_PATH)
    ic(model_fullpath)

    model = mujoco_py.load_model_from_path(model_fullpath)
    sim = mujoco_py.MjSim(model, nsubsteps=N_SUBSTEPS)

    joints = getJointInfo(model)

    # ctrl = getCtrlRange(model, i=0)

    space = ob.CompoundStateSpace()

    print(isinstance(space, ob.CompoundStateSpace))

    a = joints[0]
    print(a.range)

    # [free, ball, slide, hinge]
    t = [joints[i].type for i in range(len(joints))]
