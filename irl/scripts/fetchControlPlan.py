import argparse
import sys
import os
import pathlib
from typing import Any, Dict

from PIL import Image
import imageio

import gym
from gym.envs.robotics.utils import robot_get_obs
from gym.wrappers import FilterObservation, FlattenObservation


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from irl.scripts import ompl_utils
from irl.wrapper.fixGoal import FixGoal
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
        "-t",
        "--runtime",
        type=float,
        default=5.0,
        help="(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.",
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
            robot_qpos, robot_qvel = robot_get_obs(self.sim)
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

    robot_qpos, robot_qvel = robot_get_obs(env.sim)
    gripper_state = robot_qpos[-2:]

    grip_velp = env.sim.data.get_site_xvelp("robot0:grip") * env.dt
    gripper_vel = robot_qvel[-2:] * env.dt

    verify_obs = np.concatenate(
        [envGoal, grip_pos, gripper_state, grip_velp, gripper_vel], dtype=np.float32
    )
    assert np.all(obs == verify_obs)
    return env


def init_planning(env: gym.Env, param: Dict[str, Any]):
    # Construct the State Space we are planning in [theta, theta_dot]
    space = makeCompoundStateSpace(
        m=env.sim.model, 
        include_velocity=param["include_velocity"],
    )

    # * Since there is no deafult contorl in XML files, we need to set them manually
    control_dim = env.action_space.shape[0]  # 4
    c_space = oc.RealVectorControlSpace(space, control_dim)
    
    # Set the bounds for the control space
    c_bounds = ob.RealVectorBounds(control_dim)
    c_bounds.setLow(-1.0)
    c_bounds.setHigh(1.0)
    c_space.setBounds(c_bounds)

    # Define a simple setup class
    ss = oc.SimpleSetup(c_space)
    
    # Recover the space information
    si = ss.getSpaceInformation()
      
    # Set state validation check 
    mj_validityChecker = MujocoStateValidityChecker(si, env.sim, include_velocity=param["include_velocity"])
    ss.setStateValidityChecker(mj_validityChecker)
    
    # Set State Propagator
    mj_propagator = MujocoStatePropagator(si, env.sim, include_velocity=param["include_velocity"])
    ss.setStatePropagator(mj_propagator)
    
    # Set propagation step size 
    si.setPropagationStepSize(env.sim.model.opt.timestep)
    
    # Create a start state and a goal state
    start_state = ob.State(space)
    goal_state = ob.State(space)
    
    for i in range(param["start"].shape[0]):
        start_state[i] = param["start"][i]
        goal_state[i] = param["goal"][i]
    
    # Set the start state and goal state
    ss.setStartAndGoalStates(start_state, goal_state, 0.05)
    
    # Allocate and set the planner to the SimpleSetup
    planner = ompl_utils.allocateControlPlanner(si, plannerType=param["plannerType"])
    ss.setPlanner(planner)
    
    # Set optimization objective
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    
    # Some Sanity Check
    assert si.getStateSpace().isCompound()
    assert si.getStateSpace().isLocked()
    assert si.getStateDimension() == len(param["start"])
    for i in range(si.getStateDimension()):
        assert si.getStateSpace().getSubspaceWeight (i) == 1.0
        assert start_state[i] == param["start"][i]
        assert goal_state[i] == param["goal"][i] 
    return ss


def plan(ss: ob.SpaceInformation, param: Dict[str, Any], runtime: float):
    "Attempt to solve the problem" ""
    solved = ss.solve(runtime)
    controlPath = None
    geometricPath = None
    if solved:
        # Print the path to screen
        controlPath = ss.getSolutionPath()
        controlPath.interpolate()
        geometricPath = controlPath.asGeometric()
        geometricPath.interpolate()

        controlPath_np = np.fromstring(
            geometricPath.printAsMatrix(), dtype=float, sep="\n"
        ).reshape(-1, param["state_dim"])
        geometricPath_np = np.fromstring(
            geometricPath.printAsMatrix(), dtype=float, sep="\n"
        ).reshape(-1, param["state_dim"])
        # print("Found solution:\n%s" % path)
    else:
        print("No solution found")
    return controlPath, controlPath_np, geometricPath, geometricPath_np


if __name__ == "__main__":

    args = CLI()

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

    # Flatten and fix goal
    env = flatten_fixed_goal(env)
    env.seed(args.seed)

    # Initialize the environment
    env.reset()

    # Obtain the start state in Joint Space
    q_pos_start = env.sim.get_state().qpos
    q_vel_start = env.sim.get_state().qvel

    # Load the goal state in Joint Space from expert demonstration
    goal_data = np.load(path / "goal.npz")

    # Set the parameters of planning
    param = {
        "start": np.concatenate([q_pos_start, q_vel_start]),
        "goal": np.concatenate([goal_data["q_pos"], goal_data["q_vel"]]),
        "include_velocity": True,
        "plannerType": args.planner,
        "state_dim": 30,
    }
    
    ss = init_planning(env, param)
    
    
    si = ss.getSpaceInformation()
    space = si.getStateSpace()
    joints = getJointInfo(env.sim.model)
    
    from collections import OrderedDict
    dic = OrderedDict()
    for i in range(space.getSubspaceCount()):
        subspace = space.getSubspace(i)
        subspace_name = subspace.getName()
        dic[subspace_name] = subspace
    
    for i, (k, v) in enumerate(dic.items()):
        if isinstance(subspace, ob.RealVectorStateSpace):
            ic(i+1)
            ic(subspace.getBounds().low[0])
            ic(subspace.getBounds().high[0])
            
    ic(dic)    
    
    
    controlPath, controlPath_np, geometricPathm, geometricPath_np = plan(
        ss, param, args.runtime
    )
    ic(geometricPath_np, geometricPath_np.shape)
    
    
   
