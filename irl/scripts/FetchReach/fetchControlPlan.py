import argparse
import sys
import os
import time
import pathlib
from collections import OrderedDict
from typing import Any, Dict

from PIL import Image
import imageio

import gym
import gym.envs.robotics.utils as robotics_utils

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from irl.scripts import ompl_utils
from irl.scripts.FetchReach import fetch_utils
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


def makeCompoundStateSpaceFetchReach(
    m: PyMjModel,
    include_velocity: bool,
) -> ob.CompoundStateSpace:
    """
    Create a incomplete configuration of compound state space from the MuJoCo model.
    :param m: MuJoCo model
    :param include_velocity:
    :return: CoumpoundStateSpace
    """
    # Create the state space (optionally including velocity)
    space = ob.CompoundStateSpace()

    # Iterate over all the joints in the model
    joints = getJointInfo(m)
    vel_spaces = []

    # Add a subspace matching the topology of each joint
    next_qpos = 0
    for joint in joints:
        bounds = make_1D_VecBounds(low=joint.range[0], high=joint.range[1])
        # Check our assumptions are OK
        if joint.qposadr != next_qpos:
            raise ValueError(
                f"Joint qposadr {joint.qposadr}: Joints are not in order of qposadr."
            )
        next_qpos += 1
        # Crate an appropriate subspace based on the joint type
        if joint.type == mjtJoint.mjJNT_HINGE.value:
            if joint.limited:
                # * A hinge with limits is R^1
                joint_space = ob.RealVectorStateSpace(1)
                joint_space.setBounds(bounds)
            else:
                if joint.range[0] < joint.range[1]:
                    joint_space = ob.RealVectorStateSpace(1)
                    joint_space.setBounds(bounds)
                else:
                    joint_space = ob.SO2StateSpace()
            vel_spaces.append(ob.RealVectorStateSpace(1))

        elif joint.type == mjtJoint.mjJNT_SLIDE.value:
            joint_space = ob.RealVectorStateSpace(1)
            if joint.limited or joint.range[0] <= joint.range[1]:
                joint_space.setBounds(bounds)
            vel_spaces.append(ob.RealVectorStateSpace(1))
        else:
            raise ValueError(f"Unknown joint type {joint.type}")

        # Add the joint subspace to the compound state space
        space.addSubspace(joint_space, 1.0)

    if next_qpos != m.nq:
        raise ValueError(
            f"Total joint dimensions are not equal to nq.\nJoint dims: {next_qpos} vs nq: {m.nq}"
        )

    # Add the joint velocity subspace to the compound state space
    if include_velocity:
        for vel_space in vel_spaces:
            vel_bounds = make_1D_VecBounds(low=-1, high=1)
            vel_space.setBounds(vel_bounds)
            space.addSubspace(vel_space, 1.0)
    # DONT Lock the state space. we need to manually update the state space
    return space


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

    # * The XML file is about the robot itself which is a sharing file for all fetch env.
    space = makeCompoundStateSpaceFetchReach(
        m=env.sim.model,
        include_velocity=param["include_velocity"],
    )
    # printSubspaceInfo(space)

    # * We need to manully set the bounds of joint state and joint velocity
    subspace_lst = space.getSubspaces()

    # First 3 slide joints
    for i in range(space.getSubspaceCount()):
        if i < 3:
            joint_name = "robot0:slide" + str(i)
            bounds_i = make_1D_VecBounds(
                low=param["initial_qpos"][joint_name],
                high=param["initial_qpos"][joint_name],
                target="both",
                tolerance=1e-4,
            )
            subspace_lst[i].setBounds(bounds_i)

    # This not consit with xml file and initial q pos
    bounds_3 = make_1D_VecBounds(
        low=param["start"][3], high=param["start"][3], target="both", tolerance=1e-4
    )
    subspace_lst[3].setBounds(bounds_3)

    # Twist robot arm
    """
    I dont find any reference about this. 
    I visualize it from random samples of action space 
    maybe should be SO2
    """
    # bounds_8 = make_1D_VecBounds(
    #         low=-1.1483931076458946,
    #         high=1.8063403389140067,
    # )
    # subspace_lst[8].setBounds(bounds_8)

    # # Twist of the gripper #* I dont find any reference about this
    # bounds_12 = make_1D_VecBounds(
    #         low=0,
    #         high=0,
    #         target="both",
    #         tolerance=1e-4
    # )
    # subspace_lst[12].setBounds(bounds_12)

    # # Tips of Gripper: qpos[13], qpos[14]
    # bounds_13 = make_1D_VecBounds(
    #         low=param["block_gripper_qpos"]["robot0:l_gripper_finger_joint"],
    #         high=param["block_gripper_qpos"]["robot0:l_gripper_finger_joint"],
    #         target="both",
    #         tolerance=1e-4
    # )
    # subspace_lst[13].setBounds(bounds_13)

    # bounds_14 = make_1D_VecBounds(
    #     low=param["block_gripper_qpos"]["robot0:r_gripper_finger_joint"],
    #     high=param["block_gripper_qpos"]["robot0:r_gripper_finger_joint"],
    #     target="both",
    #     tolerance=1e-4
    # )
    # subspace_lst[14].setBounds(bounds_14)

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
    mj_validityChecker = MujocoStateValidityChecker(
        si, env.sim, include_velocity=param["include_velocity"]
    )
    ss.setStateValidityChecker(mj_validityChecker)

    # Set State Propagator
    mj_propagator = FetchReachStatePropagator(
        si, env.sim, include_velocity=param["include_velocity"]
    )
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
    assert si.getStateDimension() == len(param["start"])
    for i in range(si.getStateDimension()):
        assert si.getStateSpace().getSubspaceWeight(i) == 1.0
        assert start_state[i] == param["start"][i]
        assert goal_state[i] == param["goal"][i]
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
        geometricPath.interpolate()

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

    # Flatten and fix goal
    env = fetch_utils.flatten_fixed_goal(env)
    env.seed(args.seed)

    # Initialize the environment
    env.reset()

    # Obtain the start state in Joint Space
    q_pos_start = env.sim.get_state().qpos
    q_vel_start = env.sim.get_state().qvel

    # Load the goal state in Joint Space from expert demonstration
    goal_data = np.load(path / "goal.npz")

    # Extra configuration of FetchReach

    # https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/reach.py#L12-#L16
    initial_qpos = {
        "robot0:slide0": 0.4049,
        "robot0:slide1": 0.48,
        "robot0:slide2": 0.0,
    }
    # https://github.com/openai/gym/blob/3eb699228024f600e1e5e98218b15381f065abff/gym/envs/robotics/fetch_env.py#L77-#L79
    block_gripper_qpos = {
        "robot0:l_gripper_finger_joint": 0.0,
        "robot0:r_gripper_finger_joint": 0.0,
    }

    # Set the parameters of planning
    param = dict(
        start=np.concatenate([q_pos_start, q_vel_start]),
        goal=np.concatenate([goal_data["q_pos"], goal_data["q_vel"]]),
        include_velocity=True,
        plannerType=args.planner,
        state_dim=30,
        initial_qpos=initial_qpos,
        block_gripper_qpos=block_gripper_qpos,
    )

    ic(param["start"][:15].reshape(-1, 1))
    ic(param["start"][15:].reshape(-1, 1))

    # Setup
    ss = init_planning(env, param)

    si = ss.getSpaceInformation()
    space = si.getStateSpace()
    joints = getJointInfo(env.sim.model)

    printSubspaceInfo(space, start=param["start"])

    controlPath, _, _ = plan(ss, param, args.runtime)

    # fetch_utils.visulallzie_env(env, joint_idx=3)

    # Get controls
    controls = controlPath.getControls()
    
    control_count = controlPath.getControlCount()
    ic(control_count)
    U = [np.array([u[0], u[1], u[2], u[3]]) for u in controls]
    ic(U)
    env.reset()
    ic(env.sim.data.qpos)
    for u in U:
        obs, rew, done, info = env.step(u)

        if args.render or args.render_video:
            try:
                if args.render_video:
                    img_array = env.render(mode="rgb_array")
                    img = Image.fromarray(img_array, "RGB")
                    # images.append(img)
                else:
                    env.render(mode="human")
                    time.sleep(1)
            except KeyboardInterrupt:
                break