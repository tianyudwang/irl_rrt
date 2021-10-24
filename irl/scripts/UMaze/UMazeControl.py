import argparse
import sys
import os
import time
import pathlib

import yaml

import numpy as np
import matplotlib.pyplot as plt

import gym
import mujoco_maze
from mujoco_maze.maze_env_utils import MazeCell

import mujoco_py

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc
from ompl import geometric as og

from irl.scripts import ompl_utils
import irl.mujoco_ompl_py.mujoco_ompl_interface as mj_ompl



try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT


def convet_structure_to_num(structure):
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            structure[i][j] = structure[i][j].value
    return structure

def save2yaml(filename, data):
    # Saving hyperparams to yaml file.
    with open(path / f"{filename}_cfg.yaml", "w") as f:
        yaml.dump(data, f)

def printEnvSpace(env: gym.Env):
    print("Env space:")
    obs_space, act_space = env.observation_space, env.action_space
    obs_low, obs_high = obs_space.low, obs_space.high
    act_low, act_high = act_space.low, act_space.high
    print(f"  observation_space: {env.observation_space}")
    print(f"\tobs_low: {obs_low}")
    print(f"\tobs_high: {obs_high}")
    print(f"  env.action_space: {env.action_space}")
    print(f"\tact_low: {act_low}")
    print(f"\tact_high: {act_high}\n")

def visualize_maze(env_id: str):
    # Visualize the maze.
    env = gym.make(env_id)
    while True:
        try:
            env.render()
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

    # Observation Space and Action Space
    obs_space, act_space = env.observation_space, env.action_space
    obs_low, obs_high = obs_space.low, obs_space.high
    act_low, act_high = act_space.low, act_space.high
    printEnvSpace(env)
    
    # Visualize the maze
    if args.visual:
        visualize_maze(args.env_id)
        
    # Find associate the model
    if args.env_id.lower().find("point") != -1:
        model_fullpath = path / "point.xml"
    elif args.env_id.lower().find("ant") != -1:
        model_fullpath = path / "ant.xml"
    else:
        raise ValueError("Unknown environment")

    # Load Mujoco model
    m = mujoco_py.load_model_from_path(str(model_fullpath))

    # Raw Joint Info
    joints = mj_ompl.getJointInfo(m)
    ic(joints)
    
    # Raw Ctrl Info
    ctrls = mj_ompl.getCtrlInfo(m)
    ic(ctrls)

    # Extract the relevant information from the environment
    maze_env = env.unwrapped
    maze_task = maze_env._task
    agent_model = maze_env.wrapped_env

    # Get the maze structure
    maze_structure = maze_env._maze_structure
    # A user friendly maze structure representation
    structure_repr = np.array(
        [
            ["B", "B", "B", "B", "B"],
            ["B", "R", "E", "E", "B"],
            ["B", "B", "B", "E", "B"],
            ["B", "E", "E", "E", "B"],
            ["B", "B", "B", "B", "B"],
        ],
        dtype=object,
    )

    maze_env_config = {
        # self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]
        "goal": maze_task.goals[0].pos.tolist(),
        "goal_threshold": maze_env._task.goals[0].threshold,  # 0.6
        # "_maze_structure": maze_env._maze_structure,
        "maze_structure": structure_repr,
        "maze_size_scaling": maze_env._maze_size_scaling,
        # condition for the enviroment
        "collision": maze_env._collision,
        "_objball_collision": maze_env._objball_collision,
        "elevated": maze_env.elevated,  # False
        "blocks": maze_env.blocks,  # False
        "put_spin_near_agent": maze_env._put_spin_near_agent,  # False
        # init position should be the relative position of start position
        # self._init_positions = [(x - torso_x, y - torso_y) for x, y in self._find_all_robots()]
        "init_positons": list(maze_env._init_positions[0]),
        "init_torso_x": maze_env._init_torso_x,
        "init_torso_y": maze_env._init_torso_y,
        "xy_limits": list(
            maze_env._xy_limits()
        ),  # equavalent to env.observation_space's low and high
        
    }
    ic(maze_env_config)
    
    PointEnv_config = {
        #C++ don't recognize numpy array change to list
        "obs_high": obs_high.tolist(),
        "obs_low": obs_low.tolist(),
        "act_high": act_high.tolist(),
        "act_low": act_low.tolist(),
    }

    
    if args.dummy_setup:
        # This is a dummy setup for ease of congfiguration
        dummy_space = mj_ompl.createSpaceInformation(
            m=agent_model.sim.model,
            include_velocity=True,
        ).getStateSpace()
        if dummy_space.isCompound():
            ompl_utils.printSubspaceInfo(dummy_space, None, include_velocity=True)
    
    
    # start
    
    # State Space (A compound space include SO3 and accosicated velocity).
    # SE2 = R^2 + SO2. Should not include the bound for SO2.
    SE2_space = ob.SE2StateSpace()
    SE2_bounds = ompl_utils.make_RealVectorBounds(bounds_dim=2, low=obs_low[:2], high=obs_high[:2])
    SE2_space.setBounds(SE2_bounds)
    ompl_utils.printBounds(SE2_bounds, title="SE2 bounds")
    
    # velocity space.
    velocity_space = ob.RealVectorStateSpace(3)
    v_bounds = ompl_utils.make_RealVectorBounds(bounds_dim=3, low=obs_low[3:-1], high=obs_high[3:-1])
    velocity_space.setBounds(v_bounds)
    ompl_utils.printBounds(v_bounds, title="Velocity bounds")
    
    # Add subspace to the compound space.
    space = ob.CompoundStateSpace()
    space.addSubspace(SE2_space, 1.0)
    space.addSubspace(velocity_space, 1.0)
    
    # Lock this state space. This means no further spaces can be added as components.
    space.lock()
    
    # Create a control space and set the bounds for the control space
    cspace = oc.RealVectorControlSpace(space, 2)
    c_bounds = ompl_utils.make_RealVectorBounds(bounds_dim=2, low=act_low, high=act_high)
    cspace.setBounds(c_bounds)
    ompl_utils.printBounds(c_bounds, title="Control bounds")

    if space.isCompound():
        ompl_utils.printSubspaceInfo(space, None, include_velocity=True)
