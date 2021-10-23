import argparse
import sys
import os
import time
import pathlib
from collections import OrderedDict
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


class FetchReachStatePropagator(oc.StatePropagator):
    def __init__(
        self,
        si: oc.SpaceInformation,
        sim: MjSim,
        include_velocity: bool,
    ):
        super().__init__(si)
        self.si = si
        self.sim = sim
        self.include_velocity = include_velocity
        self.max_timestep: float = self.sim.model.opt.timestep

    def getSpaceInformation(self) -> oc.SpaceInformation:
        return self.si

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # Copy ompl state to mujoco
        copyOmplStateToMujoco(
            state, self.si, self.sim.model, self.sim.data, self.include_velocity
        )
        # Copy control
        pos_ctrl = np.empty(3)
        pos_ctrl[0] = control[0]
        pos_ctrl[1] = control[1]
        pos_ctrl[2] = control[2]
        
        gripper_ctrl = control[3]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        
        pos_ctrl *= 0.05
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        
        # Apply action to simulation.
        robotics_utils.ctrl_set_action(self.sim, action)
        robotics_utils.mocap_set_action(self.sim, action)

    def sim_duration(self, duration: float) -> None:
        steps: int = ceil(duration / self.max_timestep)
        self.sim.model.opt.timestep = duration / steps
        for _ in range(steps):
            self.sim.step()

    def canPropagateBackward(self) -> bool:
        return False

    def canSteer(self) -> bool:
        return False


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

def CLI():
    parser = argparse.ArgumentParser(description="Test the InvertedPendulum-v2 control planning")
    parser.add_argument(
        "--env_id",
        "-env",
        type=str,
        help="Envriment to interact with",
        default="InvertedPendulum-v2",
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


def printSubspaceInfo(
    space: ob.CompoundStateSpace, start: Optional[np.ndarray] = None
) -> dict:
    space_dict = OrderedDict()
    for i in range(space.getSubspaceCount()):
        subspace = space.getSubspace(i)
        name = subspace.getName()
        space_dict[name] = subspace
        low, high = None, None
        if isinstance(subspace, ob.RealVectorStateSpace):
            low, high = subspace.getBounds().low[0], subspace.getBounds().high[0]
            if start is not None:
                assert low <= start[i] <= high
        print(f"{i}: {name} \t[{low}, {high}]")
    return space_dict


def init_planning(env: gym.Env, param: Dict[str, Any]):
    # Construct the State Space we are planning in [theta, theta_dot]

    si = createSpaceInformation(
        m=env.sim.model,
        include_velocity=param["include_velocity"],
    )
    space = si.getStateSpace()
    if space.isCompound():
        printSubspaceInfo(space, param["start"])

     # Define a simple setup class
    ss = oc.SimpleSetup(si)

    # Set state validation check
    mj_validityChecker = MujocoStateValidityChecker(
        si, env.sim, include_velocity=param["include_velocity"]
    )
    ss.setStateValidityChecker(mj_validityChecker)

    # Set State Propagator
    mj_propagator = MujocoStatePropagator(
        si, env.sim, include_velocity=param["include_velocity"]
    )
    ss.setStatePropagator(mj_propagator)

    # Set propagation step size
    si.setPropagationStepSize(env.sim.model.opt.timestep)

    # Create a start state and a goal state
    start_state = ob.State(si)
    goal_state = ob.State(si)

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

    return ss


def plan(ss: ob.SpaceInformation, param: Dict[str, Any], runtime: float):
    "Attempt to solve the problem" ""
    solved = ss.solve(runtime)
    controlPath = None
    controlPath_np = None
    geometricPath = None
    geometricPath_np = None
    if solved:
        # Print the path to screen
        controlPath = ss.getSolutionPath()
        controlPath.interpolate()
        
        geometricPath = controlPath.asGeometric()
        # geometricPath.interpolate()

        geometricPath_np = np.fromstring(
            geometricPath.printAsMatrix(), dtype=float, sep="\n"
        ).reshape(-1, param["state_dim"])
        # print("Found solution:\n%s" % path)
    else:
        print("No solution found")
    return controlPath, geometricPath, geometricPath_np


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

    
    # # Set the parameters of planning
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
    ss = init_planning(env, param)

    si = ss.getSpaceInformation()
    space = si.getStateSpace()
    joints = getJointInfo(env.sim.model)


    controlPath, _, _ = plan(ss, param, args.runtime)


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