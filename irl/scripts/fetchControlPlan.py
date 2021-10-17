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
from gym.envs.robotics import rotations, robot_env, utils
from gym.wrappers import FilterObservation, FlattenObservation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

import ompl_utils
from irl.wrapper.fixGoal import FixGoal

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


def CLI():
    parser = argparse.ArgumentParser(description="Test the Hopper-v3 control planning")
    parser.add_argument(
        "--env_id",
        "-env",
        type=str,
        help="Envriment to interact with",
        default="FetchReach-v1",
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
    return args

def flatten_fixed_goal(env: gym.Env) -> gym.Env:
    """
    Filter and flatten observavtion from Dict to Box and set a fix goal state
    Before:
        obs:
            {
            'observation': array([...]]),  # (n,) depend on env 10 in FetchReach-v1
            'achieved_goal': array([...]), # (3,) # xyz pos of achieved position
            'desired_goal': array([...])  # (3,) # xyz pos of true goal position
            }
    After:
        obs:{
            "":
            "obseravtion": [desired_goal, grip_pos, gripper_state, grip_velp, gripper_vel]
                           [   (3,)       (3,)         (2,)        (3,)       (2,)   ]

            grip_pos = self.sim.data.get_site_xpos("robot0:grip")
            robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
            gripper_state = robot_qpos[-2:]
            grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
            gripper_vel = robot_qvel[-2:] * dt
        }
    :param: env
    :return flattend env where obs space in Box
    """

    # Filter the observation Dict
    env = FilterObservation(env, ["observation", "desired_goal"])

    # Convert Dict space to Box space
    env = FlattenObservation(env)

    # Fix the goal postion
    env = FixGoal(env)  # custom wrapper might need to double check

    # Sanity Check
    obs = env.reset()
    envGoal = env.goal.copy()

    grip_pos = env.sim.data.get_site_xpos("robot0:grip")

    robot_qpos, robot_qvel = utils.robot_get_obs(env.sim)
    gripper_state = robot_qpos[-2:]

    grip_velp = env.sim.data.get_site_xvelp("robot0:grip") * env.dt
    gripper_vel = robot_qvel[-2:] * env.dt

    verify_obs = np.concatenate(
        [envGoal, grip_pos, gripper_state, grip_velp, gripper_vel], dtype=np.float32
    )
    assert np.all(obs == verify_obs)
    return env

def visualize(env, random=False):
    while 1:
        try:
            if random:
                env.render()
                env.step(env.action_space.sample())
                # env.step([0,-1,0,0])
                
            else:
                for i in range(6,10):
                    ic(i)
                    for _ in range(100):
                        env.render()
                        env.sim.data.qpos[i] += 0.01
                    env.reset()
        except KeyboardInterrupt:
            break

def init_planning(param: Dict[str, Any]):
    # Construct the state space we are planning in
    # *We are planning in [theta, theta_dot]
    # Create and set bounds of theta` space.
    th_space = ob.RealVectorStateSpace(4)
    th_bounds = ob.RealVectorBounds(4)
    # th_bounds.low[0] =
    # th_bounds.high[0] =
    
    # th_bounds.low[1] =
    # th_bounds.high[1] =
    
    # th_bounds.low[2] =
    # th_bounds.high[2] =
    
    # th_bounds.low[3] =
    # th_bounds.high[3] =
    # th_space.setBounds(th_bounds)
    
    # Create and set bounds of omega space.
    omega_space = ob.RealVectorStateSpace(4)
    w_bounds = ob.RealVectorBounds(4)
    # w_bounds.low[0] =
    # w_bounds.high[0] =
    # w_bounds.low[1] =
    # w_bounds.high[1] =
    # w_bounds.low[2] =
    # w_bounds.high[2] =
    # w_bounds.low[3] =
    # w_bounds.high[3] =
    omega_space.setBounds(w_bounds)
    
     # Create compound space which allows the composition of state spaces.
    space = ob.CompoundStateSpace()
    space.addSubspace(th_space, 1.0)
    space.addSubspace(omega_space, 1.0)
    # Lock this state space. This means no further spaces can be added as components.
    space.lock()


def main():
    pass

if __name__ == "__main__":
    
    args = CLI()
    
    # Set the OMPL log level
    ompl_utils.setLogLevel(args.info)

    # Raise overflow / underflow warnings to errors
    np.seterr(all="raise")

    # Set the random seed
    ompl_utils.setRandomSeed(args.seed)

    # Create the environment
    env = gym.make(args.env_id)
    
    # Flatten and fix goal
    env = flatten_fixed_goal(env)
    env.seed(args.seed)

    # Investigate env's obs space and act space
    ic(env.observation_space)  # Box(-inf, inf, (13,), float32)
    ic(env.action_space)       # Box(-1.0, 1.0, (4,), float32)
    
    # This is same as calling env.sim.data.qpos and env.sim.data.qvel
    qpos, qvel = utils.robot_get_obs(env.sim)
    
    # useful q_pos and q_vel
    # index:  [6, 7, 8, 9,] 
    
    
    param = {
        "start": np.concatenate([qpos, qvel])[6: 12],
        "goal": env.goal  # x y z 
    }

    visualize(env, random=True)
        