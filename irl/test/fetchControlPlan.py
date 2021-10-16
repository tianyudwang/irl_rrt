import argparse
import sys
import os
import time
from typing import Any, Dict
import random

from math import pi
from functools import partial

from PIL import Image
import imageio

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

import ompl_utils

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
        0, join(dirname(dirname(dirname(abspath(__file__)))), "ompl/py-bindings")
    )
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og


def init_planning(param: Dict[str, Any]):
    # Construct the state space we are planning in
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Hopper-v3 control planning")
    parser.add_argument(
        "--env_id",
        "-env",
        type=str,
        help="Envriment to interact with",
        default="Hopper-v3",
    )
    parser.add_argument(
        "--planner",
        type=str,
        choices=["RRT", "SST", "KPIECE", "KPIECE1"],
        default="RRT",
        help="The planner to use, either RRT or SST or KPIECE",
    )
    parser.add_argument(
        "-i",
        "--info",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG. Defaults to WARN.",
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--plot", "-p", help="Render environment", action="store_true")
    parser.add_argument(
        "--render", "-r", help="Render environment", action="store_true"
    )
    parser.add_argument("--render_video", "-rv", help="Save a gif", action="store_true")

    args = parser.parse_args()
    
    # Set the OMPL log level
    ompl_utils.setLogLevel(args.info)
    
    # raise overflow / underflow warnings to errors 
    np.seterr(all="raise")

    # Set the random seed
    ompl_utils.setRandomSeed(args.seed)
    
    # Create the environment
    env = gym.make(args.env_id)
    env.seed(args.seed)
    obs = env.reset()
    ic(env.observation_space.shape)
    ic(env.action_space.shape)
    
    
    # _get_obs() ->  [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
    ic(obs)
    
    # ============================================================================================

    # ============================================================================================
    # posafter, height, ang = self.sim.data.qpos[0:3]
    ic(env.sim.data.qpos)
    # * angle in degree
    # qpos:  [pos(x), height,  angle,  angle, angle, angle]
    # limit: [inf, [0, 0.7) +-0.2   +-100(deg),  +-100(deg),  +-100(deg)]
    
    # qvel:  [+-10, +-10, +-10, +-10, +-10, +-10]
    
    ic(env.sim.data.qvel)
    ic(env.state_vector())

    # Assuming torque
    # -1.0 <= u <= 1.0 
    for i in range(10_000):
        try:
            env.render()
            
            s = env.state_vector()
            ic(s)
            qpos = s[0:6]
            qvel = s[6 :]
            qpos[5] = pi+ i
            env.set_state(qpos, qvel)
        except KeyboardInterrupt:
            
            break