import argparse
import sys
import os
import time
import pathlib
from pprint import pprint
from mujoco_maze.ant import AntEnv

import numpy as np
import matplotlib.pyplot as plt


import gym
import mujoco_maze
from mujoco_maze.agent_model import AgentModel

import mujoco_py

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


def printEnvSpace(env: gym.Env):
    print(ompl_utils.colorize("-" * 120, color="magenta"))
    print("Env space:")
    obs_space, act_space = env.observation_space, env.action_space
    obs_low, obs_high = obs_space.low, obs_space.high
    act_low, act_high = act_space.low, act_space.high

    print(f"observation_space: ")
    pprint(env.observation_space)
    print(f"obs_low:")
    pprint(obs_low)
    print(f"obs_high:")
    pprint(obs_high)

    print(ompl_utils.colorize("-" * 120, color="magenta"))
    print(f"env.action_space: {env.action_space}")
    print(f"act_low: {act_low}")
    print(f"act_high: {act_high}")
    print(ompl_utils.colorize("-" * 120, color="magenta"))


def print_state(state: ob.State, loc: str = "", color: str = "blue") -> None:
    print(ompl_utils.colorize(loc, color))
    print(
        ompl_utils.colorize(
            f"  State: {[state[0].getX(), state[0].getY(), state[0].getYaw(), state[1][0], state[1][1], state[1][2]]}\n",
            color,
        )
    )


def visualize_path(path_file: str, goal=[0, 16]):
    """
    From https://ompl.kavrakilab.org/pathVisualization.html
    """
    data = np.loadtxt(path_file)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # path
    ax.plot(data[:, 0], data[:, 1], "o-")

    ax.plot(
        data[0, 0], data[0, 1], "go", markersize=10, markeredgecolor="k", label="start"
    )
    ax.plot(
        data[-1, 0],
        data[-1, 1],
        "ro",
        markersize=10,
        markeredgecolor="k",
        label="achieved goal",
    )
    ax.plot(
        goal[0], goal[1], "bo", markersize=10, markeredgecolor="k", label="desired goal"
    )

    # Grid
    UMaze_x = np.array([-2, 10, 10, -2, -2, 6, 6, -2, -2, -2]) / 4 * 8
    UMaze_y = np.array([-2, -2, 10, 10, 6, 6, 2, 2, 2, -2]) / 4 * 8
    ax.plot(UMaze_x, UMaze_y, "r")

    plt.xlim(-8, 24)
    plt.ylim(-8, 24)
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], "o-")
    plt.show()


def makeStateSpace(
    param: dict, lock: bool = True, verbose: bool = False
) -> ob.StateSpace:
    """
    Create a state space (A compound space include SE3 + 8 RealVector space and  associated velocity).

    The free type creates a free “joint” with three translational degrees of freedom
    followed by three rotational degrees of freedom.
    The joint position is assumed to coincide with the center of the body frame.
    Thus at runtime the position and orientation data of the free joint
    correspond to the global position and orientation of the body frame.
    Free joints cannot have limits.

    In OMPL, free joint is represented by a SE3 state space.
    A state in SE(3): position = (x, y, z), quaternion = (x, y, z, w)

    Decomposed SE3 into R^3 + SO3

    Ant has 4 legs and each leg consists of two links and two joints
    ant body is a sphere with radius 0.25
    each legs has size = 2 * link_size = 2 * 0.08 = 0.16

    """

    # qpos = {R^3 + SO3 + 8D RealVector}
    # 7 dim (x,y,z qx,qy,qz,qw) + 8 dim of joint space, which is same as qpos.shape: (15,)

    # SE3 = R^3 + SO3, (SE3 space don't have constrains)

    SE3 = ob.SE3StateSpace()

    # R3 = ob.RealVectorStateSpace(3)
    R3_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=3,
        low=param["R3_low"],
        high=param["R3_high"],
    )
    SE3.setBounds(R3_bounds)

    # SO3 = ob.SO3StateSpace()  # 4 dim (x,y,z,w)

    # 8 dim of joint space
    # TODO: change this to SO2StateSpace
    # joint_space = [ob.SO2StateSpace() for _ in range(8)]
    joint_space = ob.RealVectorStateSpace(8)
    v_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=8,
        low=param["joint_bounds_low"],
        high=param["joint_bounds_high"],
    )
    joint_space.setBounds(v_bounds)

    # qvel = velocity of {R^3 + SO3 + 8D RealVector}
    # 6 dim of SE3 + 8 dim of joint velocity, which is same as qvel.shape: (14,)
    velocity_space = ob.RealVectorStateSpace(14)
    v_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=14,
        low=param["velocity_bounds_low"],
        high=param["velocity_bounds_high"],
    )
    velocity_space.setBounds(v_bounds)

    # Add subspace to the compound space.
    space = ob.CompoundStateSpace()
    space.addSubspace(SE3, 1.0)
    # for i in range(8):
    #     space.addSubspace(joint_space[i], 1.0)
    space.addSubspace(joint_space, 1.0)
    space.addSubspace(velocity_space, 1.0)

    # Lock this state space. This means no further spaces can be added as components.
    if space.isCompound() and lock:
        space.lock()

    if verbose:
        print("State Bounds Info:")
        # ompl_utils.printBounds(joint_bounds, title="Joint bounds")
        ompl_utils.printBounds(v_bounds, title="Velocity bounds")
    return space


def makeStartState(
    space: ob.StateSpace, pos: np.ndarray, bounds_low=None, bounds_high=None
) -> ob.State:
    """
    Create a start state.
    """
    start = ob.State(space)
    err = []

    # * Quaternion q = w + xi + yj + zk
    # * Mujoco is [w,x,y,z] while OMPL order is [x,y,z,w], so we swap pos[3] and pos[6]
    pos_temp = pos.copy()
    pos_temp[3], pos_temp[6] = pos_temp[6], pos_temp[3]
    for i in range(len(pos)):
        if bounds_low is not None and bounds_high is not None:
            if not (bounds_low[i] <= pos_temp[i] <= bounds_high[i]):
                err.append(i)
                print(
                    ompl_utils.colorize(
                        f"{i}: {bounds_low[i]}, {pos[i]}, {bounds_high[i]}", color="red"
                    )
                )
        start[i] = pos_temp[i]
    assert not err , f"Start state is out of bounds with index {err}"

    return start


def makeGoalState(space: ob.StateSpace, pos: np.ndarray) -> ob.State:
    """
    Create a goal state.
    """
    if isinstance(pos, np.ndarray):
        assert pos.ndim == 1
    goal = ob.State(space)
    for i in range(len(pos)):
        goal[i] = pos[i]
    return goal


def copyRealVectorState2Data(state: ob.State, data):
    for i in range(len(data)):
        data[i] = state[i]


def copyData2RealVectorState(data, state: ob.State):
    for i in range(len(data)):
        state[i] = data[i]


def copySO3State2Data(
    state,  # ob.SO3StateSpace,
    data,
) -> None:
     # * Mujoco is [w,x,y,z] while OMPL order is [x,y,z,w],
    data[0] = state.w
    data[1] = state.x
    data[2] = state.y
    data[3] = state.z


def copyData2SO3State(
    data,
    state,  # ob.SO3StateSpace,
) -> None:
    # * Mujoco is [w,x,y,z] while OMPL order is [x,y,z,w],
    state.w = data[0]
    state.x = data[1]
    state.y = data[2]
    state.z = data[3]


class MazeGoal(ob.Goal):
    """
    An OMPL Goal that satisfy when euclidean dist between state and goal under a threshold
    """

    def __init__(self, si: oc.SpaceInformation, goal: np.ndarray, threshold: float):
        super().__init__(si)
        assert goal.ndim == 1
        self.si = si
        self.goal = goal[:2]
        self.threshold = threshold

    def isSatisfied(self, state: ob.State) -> bool:
        """
        Check if the state is the goal.
        """
        # first state is R^3
        R3 = state[0]
        x, y = R3.getX(), R3.getY()
        return np.linalg.norm(self.goal - np.array([x, y])) < self.threshold


class AntStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si,
    ):
        super().__init__(si)
        self.si = si
        self.size = 0.25 + 2 * 0.08

        self.scaling = 8.0
        self.x_limits = [-4, 20]
        self.y_limits = [-4, 20]

        # Taking account the radius of the ant (body + legs) = 0.25 + 2 * 0.08
        self.Umaze_x_min = self.x_limits[0] + self.size
        self.Umaze_y_min = self.y_limits[0] + self.size
        self.Umaze_x_max = self.x_limits[1] - self.size
        self.Umaze_y_max = self.y_limits[1] - self.size

        self.counter = 0

    def isValid(self, state: ob.State) -> bool:
        R3 = state[0]
        x_pos = R3.getX()
        y_pos = R3.getY()

        # In big square contains U with point size constrained
        inSquare = all(
            [
                self.Umaze_x_min <= x_pos <= self.Umaze_x_max,
                self.Umaze_y_min <= y_pos <= self.Umaze_y_max,
            ]
        )
        if inSquare:
            inMidBlock = all(
                [
                    self.x_limits[0] <= x_pos <= (12 + self.size),
                    (4 - self.size)
                    <= y_pos
                    <= 12 + self.size,
                ]
            )
            if inMidBlock:
                valid = False
            else:
                valid = True
        # Not in big square
        else:
            valid = False

        if self.counter == 0 and not valid:
            invalid_message = f"Invalid initial states: [{x_pos}, {y_pos}]: {valid}"
            print(ompl_utils.colorize("Custom Error:", "red"))
            print(ompl_utils.colorize("\t" + "*" * 100, "red"))
            print(ompl_utils.colorize("\t" + "** " + invalid_message + " **", "red"))
            print(ompl_utils.colorize("\t" + "*" * 100, "red"))
            self.counter += 1
        # Inside empty cell and satisfiedBounds (joint velocity in range)
        return valid and self.si.satisfiesBounds(state)


class ShortestPathObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si: oc.SpaceInformation):
        super(ShortestPathObjective, self).__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        x1, y1 = s1[0].getX(), s1[0].getY()
        x2, y2 = s2[0].getX(), s2[0].getY()
        cost = np.linalg.norm([x1-x2, y1-y2])
        return ob.Cost(cost)
        # return ob.Cost(1.0)


if __name__ == "__main__":
    parser = ompl_utils.CLI()
    parser.add_argument(
        "--env_id",
        "-env",
        type=str,
        help="Envriment to interact with",
        choices=["AntUMaze-v0"],
        default="AntUMaze-v0",
    )
    args = parser.parse_args()

    # Check that time is positive
    if args.runtime <= 0:
        raise argparse.ArgumentTypeError(
            f"argument -t/--runtime: invalid choice: {args.runtime} (choose a positive number greater than 0)"
        )
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
    obs = env.reset()
    printEnvSpace(env)

    # Find associate the model
    if args.env_id.lower().find("ant") != -1:
        model_fullpath = path / "ant.xml"
    else:
        raise ValueError("Unknown environment")

    # Extract the relevant information from the environment
    maze_env = env.unwrapped
    maze_task = env.unwrapped._task
    agent_model = env.unwrapped.wrapped_env

    # Load Mujoco model
    m = mujoco_py.load_model_from_path(str(model_fullpath))

    # Raw Joint Info  (range in xml is defined in deg instead of rad)
    joints = mj_ompl.getJointInfo(m)
    # Raw Ctrl Info
    ctrls = mj_ompl.getCtrlInfo(m)
    if args.verbose:
        print(ompl_utils.colorize("-" * 120, color="magenta"))
        ompl_utils.printJointInfo(joints, title="Joint Info:")
        ompl_utils.printJointInfo(ctrls, title="Ctrls Info:")
        print(ompl_utils.colorize("-" * 120, color="magenta"))

    old_sim_state = env.unwrapped.wrapped_env.sim.get_state()
    ic(old_sim_state.qpos)
    ic(old_sim_state.qvel)
    # ic(agent_model.init_qpos)
    # ic(agent_model.init_qvel)

    # ===========================================================================
    maze_env_config = {
        # start positon is [qpos, qvel] + random noise
        "start": np.concatenate([old_sim_state.qpos, old_sim_state.qvel]),  # (15 + 14,)
        # self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]
        "goal": maze_task.goals[0].pos,
        "goal_threshold": env.unwrapped._task.goals[0].threshold,  # 0.6
        "maze_size_scaling": env.unwrapped._maze_size_scaling,
        "init_positons": list(env.unwrapped._init_positions[0]),
        "init_torso_x": env.unwrapped._init_torso_x,
        "init_torso_y": env.unwrapped._init_torso_y,
        "xy_limits": list(env.unwrapped._xy_limits()),
    }
    ic(maze_env_config)
    # ===========================================================================
    
    # These value come from the google sheet   
    
    joint_bounds_low = [
        -0.64287,
        -0.09996,
        -0.63976,
        -1.32870,
        -0.65595,
        -1.34018,
        -0.65381,
        -0.09989,
    ]
    joint_bounds_high = [
        0.6502,
        1.3356,
        0.6576,
        0.1000,
        0.6570,
        0.0997,
        0.6545,
        1.3297,
    ]

    v_xyz_lim = 5  # from the google sheet 
    v_rot_lim = 5  # from the google sheet 
    v_joint_lim = 5  
    velocity_bounds_low = (
        [-v_xyz_lim] * 3
        + [-v_rot_lim] * 3
        + [-v_joint_lim] * 8
    )

    velocity_bounds_high = (
        [v_xyz_lim] * 3
        + [v_rot_lim] * 3
        + [v_joint_lim] * 8
    )

    AntEnv_config = {
        # First Joint is Free -> SE3 = R^3 + SO3
        "R3_low": [-4, -4, 0.25], # min z takes from radius of the free joint(0.25)
        "R3_high": [20, 20, 1],   # initial z pos is 0.75
        # Rest of Joints (exclude the first free joint)
        "joint_bounds_low": joint_bounds_low,
        "joint_bounds_high": joint_bounds_high,
        # an assumption
        "velocity_bounds_low": velocity_bounds_low,
        "velocity_bounds_high": velocity_bounds_high,
        # Control Bounds
        "c_bounds_low": [ctrl.range[0] for ctrl in ctrls],
        "c_bounds_high": [ctrl.range[1] for ctrl in ctrls],
    }
    ic(AntEnv_config)

    # ===========================================================================
    # Define State Space and Control Space
    space = makeStateSpace(AntEnv_config, lock=True, verbose=args.verbose)

    if space.isCompound():
        print(ompl_utils.colorize("-" * 120, color="magenta"))
        space_dict = ompl_utils.printSubspaceInfo(space, None, include_velocity=True)
        print(ompl_utils.colorize("-" * 120, color="magenta"))
    # ===========================================================================
    # Define a simple setup class
    ss = og.SimpleSetup(space)

    # Retrieve current instance of Space Information
    si = ss.getSpaceInformation()

    # ===========================================================================
    # Set the start state and goal state
    start = makeStartState(
        space, maze_env_config["start"]
    )
    threshold = maze_env_config["goal_threshold"]  # 0.6
    # 2D Goal
    goal = MazeGoal(si, maze_env_config["goal"], threshold)
    ss.setStartState(start)
    ss.setGoal(goal)

    # ===========================================================================
    # Set State Validation Checker
    stateValidityChecker = AntStateValidityChecker(si)
    ss.setStateValidityChecker(stateValidityChecker)

    # ===========================================================================
    # Allocate and set the planner to the SimpleSetup
    planner = ompl_utils.allocateGeometricPlanner(si, plannerType="rrtstar")
    ss.setPlanner(planner)

    # Set optimization objective
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    
    
    # ===========================================================================
    # Plan
    solved = ss.solve(args.runtime)
    geometricPath = None
    if solved:
        if solved.getStatus() == ob.PlannerStatus.APPROXIMATE_SOLUTION:
            print(ompl_utils.colorize("Solution is approximate!", color="yellow"))

        elif solved.getStatus() == ob.PlannerStatus.EXACT_SOLUTION:
            print(ompl_utils.colorize("Solution is exact!", color="blue"))
        else:
            print(ompl_utils.colorize(solved.getStatus(), color="red"))(ss, args.runtime)

        # Print solution Cost
        pdef = ss.getProblemDefinition()
        geometricPath = ss.getSolutionPath()
        planner_name = ss.getPlanner().getName()
        path_length = geometricPath.length()
        cost = geometricPath.cost(pdef.getOptimizationObjective()).value()
        print(
            ompl_utils.colorize(
                f"{planner_name} found solution of path length {path_length:.4f} with an optimization objective value of {cost:.4f}",
                color="blue",
            )
        )

    # path to numpy array
    geometricPath_np = ompl_utils.path_to_numpy(geometricPath, state_dim=29)
    ic(geometricPath_np)

    if args.verbose:
        print(f"Solution:\n{geometricPath_np}\n")

    # Save the path to .txt file
    path_name = path / f"{args.env_id}_path.txt"
    with open(str(path_name), "w") as f:
        f.write(geometricPath.printAsMatrix())

    # Visualize the path
    visualize_path(path_name)

    
