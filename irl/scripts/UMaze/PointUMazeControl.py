import argparse
import sys
import os
import time
import pathlib
import math
from math import pi
import yaml

import numpy as np
import matplotlib.pyplot as plt


import gym
import mujoco_maze
from mujoco_maze.agent_model import AgentModel
from mujoco_maze.maze_env_utils import MazeCell

import mujoco_py

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

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


def print_state(state: ob.State, loc: str = "", color: str = "blue") -> None:
    print(ompl_utils.colorize(loc, color))
    print(
        ompl_utils.colorize(
            f"  State: {[state[0].getX(), state[0].getY(), state[0].getYaw(), state[1][0], state[1][1], state[1][2]]}\n",
            color,
        )
    )


def find_invalid_states(
    state: ob.State, bounds_low: list, bounds_high: list
) -> ob.State:
    assert len(bounds_low) == len(bounds_high)
    SE2_state, v_state = state[0], state[1]

    i = []
    print_state(state, "Checking state:", color="blue")
    if not (bounds_low[0] <= SE2_state.getX() <= bounds_high[0]):
        i.append(0)
    if not (bounds_low[1] <= SE2_state.getY() <= bounds_high[1]):
        i.append(1)
    if not (bounds_low[2] <= SE2_state.getYaw() <= bounds_high[2]):
        i.append(2)
    if not (bounds_low[3] <= v_state[0] <= bounds_high[3]):
        i.append(3)
    if not (bounds_low[4] <= v_state[1] <= bounds_high[4]):
        i.append(4)
    if not (bounds_low[5] <= v_state[2] <= bounds_high[5]):
        i.append(5)
    if i:
        print(ompl_utils.colorize(f"  invalid: {i}", "red"))


def visualize_path(path_file: str, goal=[0, 8]):
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
    UMaze_x = [-2, 10, 10, -2, -2, 6, 6, -2, -2, -2]
    UMaze_y = [-2, -2, 10, 10, 6, 6, 2, 2, 2, -2]
    ax.plot(UMaze_x, UMaze_y, "r")

    plt.xlim(-4, 12)
    plt.ylim(-4, 12)
    plt.legend()
    plt.show()


def makeStateSpace(
    param: dict, lock: bool = True, verbose: bool = False
) -> ob.StateSpace:
    """
    Create a state space.
    """
    # State Space (A compound space include SO3 and accosicated velocity).
    # SE2 = R^2 + SO2. Should not set the bound for SO2 since it is enfored automatically.
    SE2_space = ob.SE2StateSpace()
    SE2_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=2,
        low=param["qpos_low"],
        high=param["qpos_high"],
    )
    SE2_space.setBounds(SE2_bounds)

    # velocity space.
    velocity_space = ob.RealVectorStateSpace(3)
    v_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=3,
        low=param["qvel_low"],
        high=param["qvel_high"],
    )
    velocity_space.setBounds(v_bounds)

    # Add subspace to the compound space.
    space = ob.CompoundStateSpace()
    space.addSubspace(SE2_space, 1.0)
    space.addSubspace(velocity_space, 1.0)

    # Lock this state space. This means no further spaces can be added as components.
    if space.isCompound() and lock:
        space.lock()

    if verbose:
        print("State Bounds Info:")
        ompl_utils.printBounds(SE2_bounds, title="SE2 bounds")
        ompl_utils.printBounds(v_bounds, title="Velocity bounds")
    return space


def makeControlSpace(
    state_space: ob.StateSpace, param: dict, verbose: bool = False
) -> oc.ControlSpace:
    """
    Create a control space and set the bounds for the control space
    """
    cspace = oc.RealVectorControlSpace(state_space, 2)
    c_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=2,
        low=param["c_bounds_low"],
        high=param["c_bounds_high"],
    )
    cspace.setBounds(c_bounds)
    if verbose:
        print("Control Bounds Info:")
        ompl_utils.printBounds(c_bounds, title="Control bounds")
    return cspace


def makeStartState(space: ob.StateSpace, pos: np.ndarray) -> ob.State:
    """
    Create a start state.
    """
    if isinstance(pos, np.ndarray):
        assert pos.ndim == 1
    start = ob.State(space)
    for i in range(len(pos)):
        start[i] = pos[i]

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
        SE2_state = state[0]
        x, y = SE2_state.getX(), SE2_state.getY()
        return np.linalg.norm(self.goal - np.array([x, y])) < self.threshold


class PointStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si: oc.SpaceInformation,
    ):
        super().__init__(si)
        self.si = si
        self.size = 0.5
        self.scaling = 4.0
        self.x_limits = [-2, 10]
        self.y_limits = [-2, 10]
        
        self.Umaze_x_min = self.x_limits[0] + self.size
        self.Umaze_y_min = self.y_limits[0] + self.size
        self.Umaze_x_max = self.x_limits[1] - self.size
        self.Umaze_y_max = self.y_limits[1] - self.size
        
        '''
        ["B", "B", "B", "B", "B"]
        ["B", "R", "E", "E", "B"]
        ["B", "B", "B", "E", "B"]
        ["B", "G", "E", "E", "B"]
        ["B", "B", "B", "B", "B"]
        '''
        # [-1,0,1,2,3]
        self.counter = 0

    def isValid(self, state: ob.State) -> bool:

        SE2_state = state[0]
        assert isinstance(SE2_state, ob.SE2StateSpace.SE2StateInternal)

        x_pos = SE2_state.getX()
        y_pos = SE2_state.getY()

        # In big square contains U with point size constrained
        inSquare = all(
            [
                self.Umaze_x_min <= x_pos <= self.Umaze_x_max,
                self.Umaze_y_min <= y_pos <= self.Umaze_y_max,
            ]
        )
        if inSquare:
            # In the middle block cells
            inMidBlock = all(
                [self.x_limits[0] <= x_pos <= 6.5,# + self.size,
                 1.5<= y_pos <= 6.5,]# 2 - self.size, 6 + self.size]
            )
            if inMidBlock:
                valid = False
            else:
                valid = True
        # Not in big square
        else:
            valid = False

        # Inside empty cell and satisfiedBounds
        return valid and self.si.satisfiesBounds(state)    
    
    # def isValid(self, state: ob.State) -> bool:

    #     SE2_state = state[0]
    #     assert isinstance(SE2_state, ob.SE2StateSpace.SE2StateInternal)

    #     x_pos = SE2_state.getX()
    #     y_pos = SE2_state.getY()

    #     # In big square contains U with point size constrained
    #     inSquare = all(
    #         [
    #             -2 + self.size <= x_pos <= 10 - self.size,
    #             -2 + self.size <= y_pos <= 10 - self.size,
    #         ]
    #     )
    #     if inSquare:
    #         # In the middle block cells
    #         inMidBlock = all(
    #             [-2 <= x_pos <= 6 + self.size, 2 - self.size <= y_pos <= 6 + self.size]
    #         )
    #         if inMidBlock:
    #             valid = False
    #         else:
    #             valid = True
    #     # Not in big square
    #     else:
    #         valid = False

    #     # Inside empty cell and satisfiedBounds
    #     return valid and self.si.satisfiesBounds(state)

        # # Error Message for debugging (Invalid initial states).
        # if self.counter == 0 and not valid:
        #     invalid_message = (
        #         f"Invalid initial states: [{x_pos}, {y_pos}, {yaw_angle}]: {valid}"
        #     )
        #     print(ompl_utils.colorize("Error:", "red"))
        #     print(ompl_utils.colorize("\t" + "*" * 100, "red"))
        #     print(ompl_utils.colorize("\t" + "** " + invalid_message + " **", "red"))
        #     print(ompl_utils.colorize("\t" + "*" * 100, "red"))
        #     self.counter += 1


class PointStatePropagator(oc.StatePropagator):
    def __init__(
        self,
        si: oc.SpaceInformation,
        agent_model: AgentModel,
        velocity_limits: float,
    ):
        super().__init__(si)
        self.si = si
        self.agent_model = agent_model
        self.velocity_limits = velocity_limits

        self.bounds_low = [-2, -2, -np.pi, -12, -12, -12]
        self.bounds_high = [10, 10, np.pi, 12, 12, 12]

        # A placeholder for qpos and qvel in propagte function that don't waste tme on numpy creation
        self.qpos_temp = np.empty(3)
        self.qvel_temp = np.empty(3)

        self.counter = 0

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # Control [ballx, rot]
        assert duration == 0.02, "Propagate duration is not fixed"
        assert self.si.satisfiesBounds(state), "Input state not in bounds"
        # SE2_state: qpos = [x, y, Yaw]
        SE2_state = state[0]
        # V_state: qvel = [vx, vy, w]
        V_state = state[1]

        self.qpos_temp[0] = SE2_state.getX()
        self.qpos_temp[1] = SE2_state.getY()
        self.qpos_temp[2] = SE2_state.getYaw()

        self.qvel_temp[0] = V_state[0]
        self.qvel_temp[1] = V_state[1]
        self.qvel_temp[2] = V_state[2]

        
        self.qpos_temp[2] += control[1]
        # Check if the orientation is in [-pi, pi]
        if not(-pi <= self.qpos_temp[2] <= pi):
            # Normalize orientation to be in [-pi, pi], since it is SO2
            # * only perform this normalization when the orientation is not in [-pi, pi]
            # * This dramatically cutdown the planning time 
            self.qpos_temp[2] = ompl_utils.angle_normalize(self.qpos_temp[2])

        # Compute increment in each direction
        # using  math.sin/cos() calculate single number is much faster than numpy.sin/cos()
        ori = self.qpos_temp[2]
        self.qpos_temp[0] += math.cos(ori) * control[0]
        self.qpos_temp[1] += math.sin(ori) * control[0]

        # Clip velocity
        # *I change cbound range from [-12, 12] to [-10, 10] instead of clipping.
        # qvel = np.clip(qvel, -self.velocity_limits, self.velocity_limits)

        # copy OMPL State to Mujoco
        self.agent_model.set_state(self.qpos_temp, self.qvel_temp)

        # assume MinMaxControlDuration = 1 and frame_skip = 1
        self.agent_model.sim.step()
        next_obs = self.agent_model._get_obs()

        # Yaw angle migh be out of range [-pi, pi] after several steps.
        # Should enforced yaw angle since it should always in bounds
        if not(-pi <= next_obs[2] <= pi):
            next_obs[2] = ompl_utils.angle_normalize(next_obs[2])

        # next_obs[3:] = np.clip(next_obs[3:], -self.velocity_limits, self.velocity_limits)
        # assert -np.pi <= next_obs[2] <= np.pi, "Yaw out of bounds after mj sim step"
        # assert -10 <= next_obs[3] <= 10, "x-velocity out of bounds after mj sim step"
        # assert -10 <= next_obs[4] <= 10, "y-velocity out of bounds after mj sim step"
        # assert -10 <= next_obs[5] <= 10, "yaw-velocity out of bounds after mj sim step"

        # Copy Mujoco State to OMPL
        # next SE2_state: next_qpos = [x, y, Yaw]
        result[0].setX(next_obs[0])
        result[0].setY(next_obs[1])
        result[0].setYaw(next_obs[2])

        # next V_state: next_qvel = [vx, vy, w]
        result[1][0] = next_obs[3]
        result[1][1] = next_obs[4]
        result[1][2] = next_obs[5]

    def sim_duration(self, duration: float) -> None:
        """
        This function is not called in this example.
        """
        steps: int = math.ceil(duration / self.agent_model.sim.model.opt.timestep)
        self.agent_model.sim.model.opt.timestep = duration / steps
        for _ in range(steps):
            self.agent_model.sim.step()

    def canPropagateBackward(self) -> bool:
        return False

    def canSteer(self) -> bool:
        return False

    def satisfiesBounds(self, state: ob.State):
        if not self.si.satisfiesBounds(state):
            print_state(state, loc="\nIn StatePropagator", color="red")
            find_invalid_states(
                state, bounds_low=self.bounds_low, bounds_high=self.bounds_high
            )


class ShortestPathObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si: oc.SpaceInformation):
        super(ShortestPathObjective, self).__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        # x1, y1 = s1[0].getX(), s1[0].getY()
        # x2, y2 = s2[0].getX(), s2[0].getY()
        # cost = np.linalg.norm([x1-x2, y1-y2])
        # return ob.Cost(cost)
        return ob.Cost(1.0)


if __name__ == "__main__":
    parser = ompl_utils.CLI()
    parser.add_argument(
        "--env_id",
        "-env",
        type=str,
        help="Envriment to interact with",
        choices=["PointUMaze-v0"],
        default="PointUMaze-v0"
    )
    args = parser.parse_args()

    # Check that time is positive
    if args.runtime <= 0:
        raise argparse.ArgumentTypeError(
            "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)"
            % (args.runtime,)
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
    ic(env._max_episode_steps)
    # Initialize the environment
    env.reset()

    # Find associate the model
    if args.env_id.lower().find("point") != -1:
        model_fullpath = path / "point.xml"
    else:
        raise ValueError("Unknown environment")

    # Load Mujoco model
    m = mujoco_py.load_model_from_path(str(model_fullpath))

    # Raw Joint Info
    joints = mj_ompl.getJointInfo(m)
    # Raw Ctrl Info
    ctrls = mj_ompl.getCtrlInfo(m)
    if args.verbose:
        print(ompl_utils.colorize("-" * 120, color="magenta"))
        printEnvSpace(env)
        print(ompl_utils.colorize("-" * 120, color="magenta"))
        ompl_utils.printJointInfo(joints, title="Joint Info:")
        ompl_utils.printJointInfo(ctrls, title="Ctrls Info:")
        print(ompl_utils.colorize("-" * 120, color="magenta"))

    # Extract the relevant information from the environment
    maze_env = env.unwrapped
    maze_task = env.unwrapped._task
    # agent_model = env.unwrapped.wrapped_env
    old_sim_state = env.unwrapped.wrapped_env.sim.get_state()
    ic(old_sim_state)

    # Get the maze structure
    maze_structure = env.unwrapped._maze_structure
    # A human friendly maze structure representation(not used!)
    structure_repr = np.array(
        [
            ["B", "B", "B", "B", "B"],
            ["B", "R", "E", "E", "B"],
            ["B", "B", "B", "E", "B"],
            ["B", "G", "E", "E", "B"],
            ["B", "B", "B", "B", "B"],
        ],
        dtype=object,
    )

    maze_env_config = {
        # start positon is [qpos, qvel] + random noise
        # https://github.com/kngwyu/mujoco-maze/blob/main/mujoco_maze/point.py#L61-#L71
        "start": np.concatenate([old_sim_state.qpos, old_sim_state.qvel]),
        # self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]
        "goal": np.concatenate([maze_task.goals[0].pos, np.zeros(4)]),
        "goal_threshold": env.unwrapped._task.goals[0].threshold,  # 0.6
        # "_maze_structure": maze_env._maze_structure,
        "maze_structure": structure_repr,
        "maze_size_scaling": env.unwrapped._maze_size_scaling,
        # condition for the enviroment
        "collision": env.unwrapped._collision,
        "_objball_collision": env.unwrapped._objball_collision,
        "elevated": env.unwrapped.elevated,  # False
        "blocks": env.unwrapped.blocks,  # False
        "put_spin_near_agent": env.unwrapped._put_spin_near_agent,  # False
        # self._init_positions = [(x - torso_x, y - torso_y) for x, y in self._find_all_robots()]
        "init_positons": list(env.unwrapped._init_positions[0]),
        "init_torso_x": env.unwrapped._init_torso_x,
        "init_torso_y": env.unwrapped._init_torso_y,
        # equavalent to env.observation_space's low and high
        "xy_limits": list(env.unwrapped._xy_limits()),
    }
    ic(maze_env_config)

    qvel_max = env.unwrapped.wrapped_env.VELOCITY_LIMITS
    PointEnv_config = {
        # C++ don't recognize numpy array change to list
        # * not include qpos[3] since no bounds for SO2
        "qpos_low": env.observation_space.low.tolist()[:2],
        "qpos_high": env.observation_space.high.tolist()[:2],
        "qvel_low": [-qvel_max, -qvel_max, -qvel_max],
        "qvel_high": [qvel_max, qvel_max, qvel_max],
        "velocity_limits": qvel_max,  # 10.0
        "c_bounds_low": [ctrls[0].range[0], ctrls[1].range[0]],  # [-1, -0.25]
        "c_bounds_high": [ctrls[0].range[1], ctrls[1].range[1]],  # [1,  0.25]
    }
    ic(PointEnv_config)

    if args.dummy_setup:
        # This is a dummy setup for ease of congfiguration
        dummy_space = mj_ompl.createSpaceInformation(
            m=env.unwrapped.wrapped_env.sim.model,
            include_velocity=True,
        ).getStateSpace()
        if dummy_space.isCompound():
            ompl_utils.printSubspaceInfo(dummy_space, None, include_velocity=True)

    # ===========================================================================
    # Define State Space and Control Space
    space = makeStateSpace(PointEnv_config, lock=True, verbose=args.verbose)
    cspace = makeControlSpace(space, PointEnv_config, verbose=True)

    if space.isCompound():
        print(ompl_utils.colorize("-" * 120, color="magenta"))
        space_dict = ompl_utils.printSubspaceInfo(
            space, maze_env_config["start"], include_velocity=True
        )
        print(ompl_utils.colorize("-" * 120, color="magenta"))
    # ===========================================================================
    # Define a simple setup class
    ss = oc.SimpleSetup(cspace)

    # Retrieve current instance of Space Information
    si = ss.getSpaceInformation()

    # Retrieve current instance of the problem definition
    pdef = ss.getProblemDefinition()

    # ===========================================================================
    # Set the start state and goal state
    start = makeStartState(space, maze_env_config["start"])
    # threshold = 0.05
    threshold = maze_env_config["goal_threshold"]  # in 6D

    if args.custom_goal:
        # 2D Goal
        goal = MazeGoal(si, maze_env_config["goal"], threshold)
        ss.setStartState(start)
        # *keep with ss since pdef.setGoal(goal) returns a copy will not affect the planning.
        ss.setGoal(goal)

    else:
        goal = makeGoalState(space, maze_env_config["goal"])
        # Set the start and goal states with threshold
        #! In 8D case the threshold is based distance of all 8 dimension!!
        ss.setStartAndGoalStates(start, goal, threshold)

    print(f"Start: {maze_env_config['start']}")
    print(f"Goal: {maze_env_config['goal']}")
    # ===========================================================================
    # Set State Validation Checker
    stateValidityChecker = PointStateValidityChecker(si)
    ss.setStateValidityChecker(stateValidityChecker)

    # *There is no need to set Motion Validation Checker in contorl planning
    # * setMotionValidator() seems that can only be called from si
    # * oc.spaceInformation.propagateWhileValid() is taking carte of coillisons

    # ===========================================================================
    # Set State Propagator
    propagator = PointStatePropagator(
        si,
        env.unwrapped.wrapped_env,
        velocity_limits=PointEnv_config["velocity_limits"],
    )
    ss.setStatePropagator(propagator)

    # Set propagator step size
    si.setPropagationStepSize(
        env.unwrapped.wrapped_env.sim.model.opt.timestep
    )  # deafult 0.05 in ompl. 0.02 in Mujoco
    si.setMinMaxControlDuration(minSteps=1, maxSteps=1)
    # ===========================================================================
    # Allocate and set the planner to the SimpleSetup
    planner = ompl_utils.allocateControlPlanner(si, plannerType=args.planner)
    ss.setPlanner(planner)

    # Set optimization objective
    # ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    objective = ShortestPathObjective(si)
    ss.setOptimizationObjective(objective)

    # ===========================================================================
    """
    Possible Error:
        (1) Error:   RRT: There are no valid initial states!
            - Check if the start state is in bounds
            - Check if the start state in stateValidityChecker.isvalid()
            - Check if there is noise add to start state
            - Check the state valid condition
        (2)  bounds not satisfied in propogator.propagate()
            - Check if there is a clip in control during the calculation. (should enforce it as the cbounds)
            - ompl check first state and call propagate() first and then pass to isValid()
                It is OK that `result` state is not in bounds. It will be invalid in stateValidityChecker.isvalid()
            - one loop wrap angle cannot properly warp the angle > abs(3* pi)
            - Remember to check the angle after sim.step. It might be out of [-pi, pi].
                Enforce it to be bounds if it is a SO2 state. Otherwise, isVlaid will take care of the rest of them.
        (3) MotionValidator not working
            - There is a propagateWhileValid() which calls isValid() and propagate() alternatively.
            - This function stop if a collision is found and return the previous number of steps,
                which actually performed without collison.
            - Since the propagatorStepSize is relatively small.
            This should be good enough to ensure the motion is valid.
        (4) Propagate Duration
            - Sync the duration with num of mujoco sim step 
    """

    # Plan
    controlPath, geometricPath = ompl_utils.plan(ss, args.runtime)

    # Make the path such that all controls are applied for a single time step (computes intermediate states)
    controlPath.interpolate()

    # path to numpy array
    geometricPath_np = ompl_utils.path_to_numpy(geometricPath, state_dim=6)

    if args.verbose:
        print(f"Solution:\n{geometricPath_np}\n")

    # Save the path to .txt file
    path_name = path / f"{args.env_id}_path.txt"
    with open(str(path_name), "w") as f:
        f.write(geometricPath.printAsMatrix())

    # Visualize the path
    visualize_path(path_name)

    # retrive controls
    ompl_controls = controlPath.getControls()
    control_count = controlPath.getControlCount()

    # Ensure we have the same start position
    env.unwrapped.wrapped_env.sim.set_state(old_sim_state)

    controls = np.asarray([[u[0], u[1]] for u in ompl_controls])
    ic(controls.shape)

    confrim_render = input("Do you want to render the environment ([y]/n)? ")
    # Render the contol path in gym environment
    for u in controls:
        qpos = env.unwrapped.wrapped_env.sim.data.qpos
        qvel = env.unwrapped.wrapped_env.sim.data.qvel
        # print(f"qpos: {qpos}, qvel: {qvel}")

        obs, rew, done, info = env.step(u)
        if args.render and confrim_render.lower() == "y":
            try:
                if args.render_video:
                    img_array = env.render(mode="rgb_array")
                else:
                    env.render(mode="human")
                    time.sleep(0.5)
            except KeyboardInterrupt:
                break
        if done:
            ic(info)
            print(ompl_utils.colorize("Reach Goal. Success!!", color="green"))
            break
    env.close()
