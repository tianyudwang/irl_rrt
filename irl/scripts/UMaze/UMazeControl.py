import sys
import os
import time
import pathlib

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


def visualize_path(path_file: str):
    """
    From https://ompl.kavrakilab.org/pathVisualization.html
    """
    data = np.loadtxt(path_file)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot(data[:, 1], data[:, 2], "o-")
    ax.plot(
        data[0, 1], data[0, 2], "go", markersize=10, markeredgecolor="k", label="start"
    )
    ax.plot(
        data[-1, 1], data[-1, 2], "ro", markersize=10, markeredgecolor="k", label="goal"
    )
    plt.legend()
    plt.show()


"""
#idx 0   1     2    3    4
-4 ["B",  "B",  "B",  "B",  "B"],
0  ["B",  "R",  "E",  "E",  "B"],
4  ["B",  "B",  "B",  "E",  "B"],
8  ["B",  "G",  "E",  "E",  "B"],
12 ["B",  "B",  "B",  "B",  "B"],
   -4      0     4     8       12
"""


class PointStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si: oc.SpaceInformation,
    ):
        super().__init__(si)
        self.si = si
        self.counter = 0
        self.noise_high = 0.1

    def isValid(self, state: ob.State) -> bool:

        SE2_state = state[0]
        assert isinstance(SE2_state, ob.SE2StateSpace.SE2StateInternal)

        # import ipdb; ipdb.set_trace()

        x_pos = SE2_state.getX()
        y_pos = SE2_state.getY()
        yaw_angle = SE2_state.getYaw()
        valid = False

        if not self.si.satisfiesBounds(state):
            return False

        # * The problem is that we define the state space as a grid of cells.
        # * initial_qpos = [0, 0, 0], initial_qvel = [0, 0, 0]
        # * However, the actual qpos = initial_qpos + noise (Uniform [low=-0.1, high=0.1])
        # *          the actual qvel = initial_qvel + noise (standard normal * 0.1)
        # * Leading to invalid initial states (sometimes).
        if -self.noise_high <= x_pos <= 10:  # This should always be true.
            if 8 <= x_pos <= 10 and 4 <= y_pos <= 8:
                valid = True
            else:
                if -self.noise_high <= y_pos <= 4:
                    valid = True
                elif 8 <= y_pos <= 10:
                    valid = True

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

        return valid


class MotionValidator(ob.MotionValidator):
    def __init__(self, si, collision):
        super().__init__(si)
        self.si = si
        self.collision = collision

    def checkMotion(self, s1: ob.State, s2: ob.State) -> bool:
        # conver s1 and s2 to 1D numpy arrays
        old_pos = np.array([s1[0].getX(), s1[0].getY()])
        new_pos = np.array([s2[0].getX(), s2[0].getY()])

        col = self.collision.detect(old_pos, new_pos)
        return col is None


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

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:

        assert self.si.satisfiesBounds(state), "Input state not in bounds"
        # SE2_state: qpos = [x, y, Yaw]
        SE2_state = state[0]
        # V_state: qvel = [vx, vy, w]
        V_state = state[1]

        qpos = np.array([SE2_state.getX(), SE2_state.getY(), SE2_state.getYaw()])
        qvel = np.array([V_state[0], V_state[1], V_state[2]])

        qpos[2] += control[1] 
        
        # Clip orientation
        # ! deperacated : a single clip can not constrain the angle larger than 3 pi
        # if qpos[2] < -np.pi:
        #     qpos[2] += np.pi * 2
        # elif np.pi < qpos[2]:
        #     qpos[2] -= np.pi * 2
        qpos[2] = ompl_utils.angle_normalize(qpos[2])
        ori = qpos[2]
        assert (-np.pi <= ori <= np.pi)
        
        # Compute increment in each direction
        qpos[0] += np.cos(ori) * control[0]
        qpos[1] += np.sin(ori) * control[0]

        # I change cbound range between [-10, 10] instead of clipping.
        # Clip velocity  #? This is contradict to the control range
        # qvel = np.clip(qvel, -self.velocity_limits, self.velocity_limits)

        self.agent_model.set_state(qpos, qvel)
        for _ in range(0, self.agent_model.frame_skip):
            self.agent_model.sim.step()
        next_obs = self.agent_model._get_obs()

        # TODO: investigate duration

        # next SE2_state: next_qpos = [x, y, Yaw]
        result[0].setX(next_obs[0])
        result[0].setY(next_obs[1])
        result[0].setYaw(next_obs[2])

        # next V_state: next_qvel = [vx, vy, w]
        result[1][0] = next_obs[3]
        result[1][1] = next_obs[4]
        result[1][2] = next_obs[5]

        if not self.si.satisfiesBounds(result):
            print_state(result, loc="\nIn StatePropagator", color='red')
            find_invalid_states(result, bounds_low=self.bounds_low, bounds_high=self.bounds_high)

    # def sim_duration(self, duration: float) -> None:
    #     steps: int = np.ceil(duration / self.max_timestep)
    #     self.sim.model.opt.timestep = duration / steps
    #     for _ in range(steps):
    #         self.sim.step()

    def canPropagateBackward(self) -> bool:
        return False

    def canSteer(self) -> bool:
        return False


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

    # Get the maze structure
    maze_structure = env.unwrapped._maze_structure
    # A human friendly maze structure representation
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
        "start": env.unwrapped.wrapped_env._get_obs(),
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
        "xy_limits": list(
            env.unwrapped._xy_limits()
        ),  # equavalent to env.observation_space's low and high
    }
    ic(maze_env_config)

    VELOCITY_LIMITS = env.unwrapped.wrapped_env.VELOCITY_LIMITS
    PointEnv_config = {
        # C++ don't recognize numpy array change to list
        "obs_high": env.observation_space.high.tolist(),
        "obs_low": env.observation_space.low.tolist(),
        "act_high": env.action_space.high.tolist(),
        "act_low": env.action_space.low.tolist(),
        "velocity_limits": VELOCITY_LIMITS,  # 10.0
        "c_bounds_low": [-VELOCITY_LIMITS, -VELOCITY_LIMITS],
        "c_bounds_high": [VELOCITY_LIMITS, VELOCITY_LIMITS],
    }

    if args.dummy_setup:
        # This is a dummy setup for ease of congfiguration
        dummy_space = mj_ompl.createSpaceInformation(
            m=env.unwrapped.wrapped_env.sim.model,
            include_velocity=True,
        ).getStateSpace()
        if dummy_space.isCompound():
            ompl_utils.printSubspaceInfo(dummy_space, None, include_velocity=True)

    # State Space (A compound space include SO3 and accosicated velocity).
    # SE2 = R^2 + SO2. Should not set the bound for SO2 since it is enfored automatically.
    SE2_space = ob.SE2StateSpace()
    SE2_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=2,
        low=PointEnv_config["obs_low"][:2],
        high=PointEnv_config["obs_high"][:2],
    )
    SE2_space.setBounds(SE2_bounds)
    # print("Bounds Info:")
    # ompl_utils.printBounds(SE2_bounds, title="SE2 bounds")

    # velocity space.
    velocity_space = ob.RealVectorStateSpace(3)
    v_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=3,
        low=PointEnv_config["obs_low"][3:-1],
        high=PointEnv_config["obs_high"][3:-1],
    )
    velocity_space.setBounds(v_bounds)
    # ompl_utils.printBounds(v_bounds, title="Velocity bounds")

    # Add subspace to the compound space.
    space = ob.CompoundStateSpace()
    space.addSubspace(SE2_space, 1.0)
    space.addSubspace(velocity_space, 1.0)

    # Lock this state space. This means no further spaces can be added as components.
    space.lock()

    # Create a control space and set the bounds for the control space
    cspace = oc.RealVectorControlSpace(space, 2)
    c_bounds = ompl_utils.make_RealVectorBounds(
        bounds_dim=2,
        low=PointEnv_config["c_bounds_low"],
        high=PointEnv_config["c_bounds_high"],
    )
    cspace.setBounds(c_bounds)
    # ompl_utils.printBounds(c_bounds, title="Control bounds")

    # Set the start state and goal state
    start = ob.State(space)
    goal = ob.State(space)
    ic(maze_env_config["start"])
    ic(maze_env_config["goal"])

    for i in range(maze_env_config["start"].shape[0]):
        start[i] = maze_env_config["start"][i]
        goal[i] = maze_env_config["goal"][i]

    # Define a simple setup class
    ss = oc.SimpleSetup(cspace)

    # Set the start and goal states with threshold
    # ss.setStartAndGoalStates(start, goal, maze_env_config["goal_threshold"])
    ss.setStartAndGoalStates(start, goal, 0.05)

    # retrieve the Space Information from the SimpleSetup
    si = ss.getSpaceInformation()

    # Set state validation check
    stateValidityChecker = PointStateValidityChecker(si)
    ss.setStateValidityChecker(stateValidityChecker)

    # State propagator
    propagator = PointStatePropagator(
        si,
        env.unwrapped.wrapped_env,
        velocity_limits=PointEnv_config["velocity_limits"],
    )
    ss.setStatePropagator(propagator)

    # Set propagator step size
    si.setPropagationStepSize(
        env.unwrapped.wrapped_env.sim.model.opt.timestep
    )  # deafult 0.05, 0.02 in Mujoco

    # Allocate and set the planner to the SimpleSetup
    planner = ompl_utils.allocateControlPlanner(si, plannerType=args.planner)
    ss.setPlanner(planner)

    # Set optimization objective
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    """
    Possible Error:
        (1) Error:   RRT: There are no valid initial states!
            - Check if the start state is in bounds
            - Check if the start state in stateValidityChecker.isvalid()
            - CHeck the state valid condition
        (2)  bounds not satisfied in propogator.propagate()
            - Check if there is a clip in control during the calculation. (should enforce it as the cbounds)
            - ompl check first state and call propagate() first and then pass to isValid()
                It is OK that `result` state is not in bounds. It will be invalid in stateValidityChecker.isvalid()
            - one loop wrap angle cannot properly warp the angle > abs(3* pi) 
    """

    if space.isCompound():
        print(ompl_utils.colorize("-" * 120, color="magenta"))
        space_dict = ompl_utils.printSubspaceInfo(
            space, maze_env_config["start"], include_velocity=True
        )
        print(ompl_utils.colorize("-" * 120, color="magenta"))

    # Plan
    controlPath, geometricPath = ompl_utils.plan(ss, args.runtime)

    controlPath.interpolate()

    # path to numpy array
    # controlPath_np = ompl_utils.path_to_numpy(controlPath, state_dim=6)
    geometricPath_np = ompl_utils.path_to_numpy(geometricPath, state_dim=6)

    ic(geometricPath_np)

    # Save the path to .txt file
    path_name = path / f"{args.env_id}_path.txt"
    with open(str(path_name), "w") as f:
        f.write(geometricPath.printAsMatrix())

    # Visualize the path
    visualize_path(path_name)

    # Render the path in gym environment
