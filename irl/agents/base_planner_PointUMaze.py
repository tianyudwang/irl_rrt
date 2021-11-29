from math import pi, sin, cos
from typing import Union, Tuple


import numpy as np
from mujoco_maze.agent_model import AgentModel

from irl.agents.planner_utils import (
    allocateControlPlanner,
    allocateGeometricPlanner,
    angle_normalize,
    baseUMazeGoalState,
    baseUMazeStateValidityChecker,
    copyData2SE2State,
    copySE2State2Data,
    make_RealVectorBounds,
    path_to_numpy,
)
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc


def visualize_path(data: np.ndarray, goal=[0, 8]):
    from matplotlib import pyplot as plt

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


def getIRLCostObjective(si, cost_fn) -> ob.OptimizationObjective:
    return IRLCostObjective(si, cost_fn)


class IRLCostObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super(IRLCostObjective, self).__init__(si)
        self.cost_fn = cost_fn

        self.s1_data = np.empty(6, dtype=np.float32)
        self.s2_data = np.empty(6, dtype=np.float32)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        # pos and rot
        copySE2State2Data(data=self.s1_data, state=s1[0])
        copySE2State2Data(data=self.s2_data, state=s2[0])

        for i in range(3):
            self.s1_data[i] = s1[1][i]
            self.s2_data[i] = s2[1][i]

        # x1, y1, yaw1 = s1[0].getX(), s1[0].getY(), s1[0].getYaw()
        # x2, y2, yaw2 = s2[0].getX(), s2[0].getY(), s2[0].getYaw()

        # linear and angular velocities
        # x1_dot, y1_dot, yaw1_dot = s1[1][0], s1[1][1], s1[1][2]
        # x2_dot, y2_dot, yaw2_dot = s2[1][0], s2[1][1], s2[1][2]

        # TODO:
        # ? Should this be 6D?
        # ? Do we need to ensure s1 and s2 has dtype float32?
        c = self.cost_fn(s1, s2)
        return ob.Cost(c)


class PointUMazeGoalState(baseUMazeGoalState):
    def __init__(self, si: ob.SpaceInformation, goal: np.ndarray, threshold: float):
        super(PointUMazeGoalState, self).__init__(si, goal, threshold)

    def sampleGoal(self, state: ob.State) -> None:
        state[0].setXY(self.goal[0], self.goal[1])
        # following are dummy values needed for properly setting the goal state
        # only using x and y to calculate the distance to the goal
        state[0].setYaw(0)
        for i in range(3):
            state[1][i] = 0


class PointStateValidityChecker(baseUMazeStateValidityChecker):
    def __init__(
        self,
        si: Union[oc.SpaceInformation, ob.SpaceInformation],
        size: float = 0.5,
        scaling: float = 4,
    ):
        super(PointStateValidityChecker, self).__init__(si, size, scaling)


class PointStatePropagator(oc.StatePropagator):
    def __init__(
        self,
        si: oc.SpaceInformation,
        agent_model: AgentModel,
        velocity_limits: float = 10,
    ):
        super().__init__(si)
        self.si = si
        self.agent_model = agent_model
        self.velocity_limits = velocity_limits

        # A placeholder for qpos and qvel in propagte function that don't waste tme on numpy creation
        self.qpos_temp = np.empty(3)
        self.qvel_temp = np.empty(3)

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # Control [ballx, rot]
        assert self.si.satisfiesBounds(state), "Input state not in bounds"
        # SE2_state: qpos = [x, y, Yaw]
        SE2_state = state[0]
        # V_state: qvel = [vx, vy, w]
        V_state = state[1]

        copySE2State2Data(state=SE2_state, data=self.qpos_temp)

        for i in range(3):
            self.qvel_temp[i] = V_state[i]

        self.qpos_temp[2] += control[1]
        # Normalize orientation to be in [-pi, pi], since it is SO2
        if not (-pi <= self.qpos_temp[2] <= pi):
            self.qpos_temp[2] = angle_normalize(self.qpos_temp[2])

        # Compute increment in each direction
        ori = self.qpos_temp[2]
        self.qpos_temp[0] += cos(ori) * control[0]
        self.qpos_temp[1] += sin(ori) * control[0]

        # Clip velocity enforced in cbound

        # copy OMPL State to Mujoco
        self.agent_model.set_state(self.qpos_temp, self.qvel_temp)

        # assume MinMaxControlDuration = 1 and frame_skip = 1
        self.agent_model.sim.step()
        next_obs = self.agent_model._get_obs()

        # Yaw angle migh be out of range [-pi, pi] after several steps.
        # Should enforced yaw angle since SO2 should always in bounds
        if not (-pi <= next_obs[2] <= pi):
            next_obs[2] = angle_normalize(next_obs[2])

        # Copy Mujoco State to OMPL
        # next SE2_state: next_qpos = [x, y, Yaw]

        copyData2SE2State(state=result[0], data=next_obs[:3])

        # next V_state: next_qvel = [vx, vy, w]
        for i in range(3):
            result[1][i] = next_obs[3 + i]

    def canPropagateBackward(self) -> bool:
        return False

    def canSteer(self) -> bool:
        return False


class BasePlannerPointUMaze:
    def __init__(self, env, use_control=False, log_level=0):
        """
        other_wrapper(<TimeLimit<MazeEnv<PointUMaze-v0>>>)  --> <MazeEnv<PointUMaze-v0>>
        """
        self.agent_model = env.unwrapped.wrapped_env
        # space information
        self.state_dim = 6
        self.control_dim = 2

        # state bound
        state_high = np.array([10, 10, pi, 10, 10, 10])
        state_low = np.array([-2, -2, pi, -10, -10, -10])

        # control bound
        self.control_high = np.array([1, 0.25])
        self.control_low = -self.control_high

        # goal position and goal radius
        self.goal_pos = np.array([0, 8])
        self.threshold = 0.6

        self.qpos_low = state_low[:2]
        self.qpos_high = state_high[:2]
        self.qvel_low = state_low[3:]
        self.qvel_high = state_high[3:]

        self.space: ob.StateSpace = None
        self.cspace: oc.ControlSpace = None
        self.init_simple_setup(use_control, log_level)

    @property
    def PropagStepSize(self) -> float:
        return self.agent_model.sim.model.opt.timestep

    def makeStateSpace(self, lock: bool = True) -> ob.StateSpace:
        """
        Create a state space.
        """
        # State Space (A compound space include SO3 and accosicated velocity).
        # SE2 = R^2 + SO2. Should not set the bound for SO2 since it is enfored automatically.
        SE2_space = ob.SE2StateSpace()
        SE2_bounds = make_RealVectorBounds(
            bounds_dim=2,
            low=self.qpos_low,
            high=self.qpos_high,
        )
        SE2_space.setBounds(SE2_bounds)

        # velocity space.
        velocity_space = ob.RealVectorStateSpace(3)
        v_bounds = make_RealVectorBounds(
            bounds_dim=3,
            low=self.qvel_low,
            high=self.qvel_high,
        )
        velocity_space.setBounds(v_bounds)

        # Add subspace to the compound space.
        space = ob.CompoundStateSpace()
        space.addSubspace(SE2_space, 1.0)
        space.addSubspace(velocity_space, 1.0)

        # Lock this state space. This means no further spaces can be added as components.
        if space.isCompound() and lock:
            space.lock()
        return space

    def makeControlSpace(self, state_space: ob.StateSpace) -> oc.ControlSpace:
        """
        Create a control space and set the bounds for the control space
        """
        cspace = oc.RealVectorControlSpace(state_space, self.control_dim)
        c_bounds = make_RealVectorBounds(
            bounds_dim=self.control_dim,
            low=self.control_low,
            high=self.control_high,
        )
        cspace.setBounds(c_bounds)
        return cspace

    def makeStartState(self, s0: np.ndarray) -> ob.State:
        """
        Create a start state.
        """
        if isinstance(s0, np.ndarray):
            assert s0.ndim == 1
        start_state = ob.State(self.space)
        for i in range(len(s0)):
            # * Copy an element of an array to a standard Python scalar
            # * to ensure C++ can recognize it.
            start_state[i] = s0[i].item()
        return start_state

    def init_simple_setup(self, use_control: bool = False, log_level: int = 0):
        """
        Initialize an ompl::control::SimpleSetup instance
            or ompl::geometric::SimpleSetup instance. if use_control is False.
        """
        assert isinstance(log_level, int) and 0 <= log_level <= 2
        log = {
            0: ou.LOG_WARN,
            1: ou.LOG_INFO,
            2: ou.LOG_DEBUG,
        }
        # Set log to warn/info/debug
        ou.setLogLevel(log[log_level])
        print(f"Set OMPL LOG_LEVEL to {log[log_level]}")

        # Define State Space
        self.space = self.makeStateSpace()

        if use_control:
            self.cspace = self.makeControlSpace(self.space)
            # Define a simple setup class
            self.ss = oc.SimpleSetup(self.cspace)
            # Retrieve current instance of Space Information
            self.si = self.ss.getSpaceInformation()

            # Set State Propagator
            propagator = PointStatePropagator(self.si, self.agent_model, 10)
            self.ss.setStatePropagator(propagator)

            # Set propagator step size
            self.si.setPropagationStepSize(self.PropagStepSize)  # 0.02 in Mujoco
            self.si.setMinMaxControlDuration(minSteps=1, maxSteps=1)
        else:
            # Define a simple setup class
            self.ss = og.SimpleSetup(self.space)
            # Retrieve current instance of Space Information
            self.si = self.ss.getSpaceInformation()

        # Set State Validation Checker
        stateValidityChecker = PointStateValidityChecker(self.si)
        self.ss.setStateValidityChecker(stateValidityChecker)
        # Set the goal state
        goal_state = PointUMazeGoalState(self.si, self.goal_pos, self.threshold)
        self.ss.setGoal(goal_state)

        # * planner not set in BasePlanner

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)

    def clearDataAndSetStartState(self, s0: np.ndarray):
        """
        Clear previous planning computation data, does not affect settings and start/goal
        And set a new start state.
        """
        self.ss.clear()
        # Reset the start state
        start = self.makeStartState(s0)
        self.ss.setStartState(start)

    def control_plan(
        self, start_state: np.ndarray, solveTime: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform control planning for a specified amount of time.
        """
        self.clearDataAndSetStartState(start_state)

        solved = self.ss.solve(solveTime)
        if solved:
            control_path = self.ss.getSolutionPath()
            geometricPath = control_path.asGeometric()
            states = path_to_numpy(
                geometricPath, state_dim=self.state_dim, dtype=np.float32
            )
            controls = np.asarray(
                [[u[0], u[1]] for u in control_path.getControls()], dtype=np.float32
            )
            return states, controls
        else:
            raise ValueError("OMPL is not able to solve under current cost function")

    def geometric_plan(
        self, start_state: np.ndarray, solveTime: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform geometric planning for a specified amount of time.
        """
        self.clearDataAndSetStartState(start_state)

        solved = self.ss.solve(solveTime)
        if solved:
            geometricPath = self.ss.getSolutionPath()
            states = path_to_numpy(geometricPath, self.state_dim, dtype=np.float32)
            controls = None
            # return the states and controls(which is None in og plannig)
            return states, controls
        else:
            raise ValueError("OMPL is not able to solve under current cost function")


class ControlPlanner(BasePlannerPointUMaze):
    """
    Control planning using oc.planner
    """

    def __init__(self, env, plannerType: str, log_level: int = 0):
        super(ControlPlanner, self).__init__(env, True, log_level)
        self.init_planner(plannerType)

    def init_planner(self, plannerType: str):
        # Set planner
        planner = allocateControlPlanner(self.si, plannerType)
        self.ss.setPlanner(planner)

    def plan(
        self, start_state: np.ndarray, solveTime: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().control_plan(start_state, solveTime)


class GeometricPlanner(BasePlannerPointUMaze):
    """
    Geometric planning using og.planner
    """

    def __init__(self, env, plannerType: str, log_level: int = 0):
        super(GeometricPlanner, self).__init__(env, False, log_level)
        self.init_planner(plannerType)

    def init_planner(self, plannerType: str):
        # Set planner
        planner = allocateGeometricPlanner(self.si, plannerType)
        self.ss.setPlanner(planner)

    def plan(
        self, start_state: np.ndarray, solveTime: float = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().geometric_plan(start_state, solveTime)


class DummyStartStateValidityChecker:
    def __init__(
        self,
    ):
        # Point radius
        self.size = 0.5
        # self.x_lim_low = self.y_lim_low = -2
        # self.x_lim_high = self.y_lim_high = 10

        self.Umaze_x_min = self.Umaze_y_min = -2 + self.size
        self.Umaze_x_max = self.Umaze_y_max = 10 - self.size

    def isValid(self, state: np.ndarray) -> bool:

        #

        x_pos = state[0]
        y_pos = state[1]

        # In big square contains U with point size constrained
        inSquare = all(
            [
                self.Umaze_x_min <= x_pos <= self.Umaze_x_max,
                self.Umaze_y_min <= y_pos <= self.Umaze_y_max,
            ]
        )
        if inSquare:
            # In the middle block cells
            inMidBlock = (-2 <= x_pos <= 6.5) and (1.5 <= y_pos <= 6.5)
            if inMidBlock:
                valid = False
            else:
                valid = True
        # Not in big square
        else:
            valid = False

        # Inside empty cell and satisfiedBounds
        return valid


def test_100(use_control_plan, plannerType, visualize=False):
    assert plannerType in ["rrt", "sst", "rrtstar", "prmstar"]

    import random
    import gym
    import mujoco_maze
    from tqdm import tqdm

    print(f"\nStart testing with {plannerType}...")
    env = gym.make("PointUMaze-v0")
    print("max_ep_len:", env.spec.max_episode_steps)

    if use_control_plan:
        planner = ControlPlanner(env, plannerType, log_level=1)
    else:
        planner = GeometricPlanner(env, plannerType, log_level=1)

    seed = 0
    ou.RNG(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    start_checker = DummyStartStateValidityChecker()

    err = 0
    excced = 0
    plan_success = 0
    if use_control_plan:

        for i in tqdm(range(100), dynamic_ncols=True):
            obs = env.reset()

            while True:
                x, y = np.random.uniform(-1.5, 9.5, size=2)
                start = np.array([x, y])
                valid = start_checker.isValid(state=start)
                if valid:
                    obs[:2] = start
                    break

            state, controls = planner.plan(start_state=obs[:-1], solveTime=5)
            if visualize:
                visualize_path(state)
            # Ensure we have the same start position
            env.unwrapped.wrapped_env.set_state(obs[:3], obs[3:-1])
            for i, u in enumerate(controls):
                obs, rew, done, info = env.step(u)

                if done:
                    print("Reach Goal. Success!!")
                    plan_success += 1
                    break
                if i == len(controls) - 1:
                    print(info)
                    print("EXCEED max_ep_len")
                    excced += 1
            if not done:
                err += 1
        print("err", err)
        print("excced", err)
    else:
        obs = env.reset()
        while True:
            x, y = np.random.uniform(-1.5, 9.5, size=2)
            start = np.array([x, y])
            valid = start_checker.isValid(state=start)
            if valid:
                obs[:2] = start
                break
        state, _ = planner.plan(start_state=obs[:-1], solveTime=5)
        if visualize:
            visualize_path(state)


if __name__ == "__main__":
    # Test passed
    # test_100(use_control_plan=True, plannerType="rrt")
    # test_100(use_control_plan=True, plannerType="sst")
    # test_100(use_control_plan=False, plannerType="rrtstar", visualize=True)
    test_100(use_control_plan=False, plannerType="prmstar", visualize=True)
