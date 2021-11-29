from itertools import chain
from typing import Tuple, Union


import numpy as np
from mujoco_maze.agent_model import AgentModel

from irl.agents.planner_utils import (
    allocateControlPlanner,
    allocateGeometricPlanner,
    make_RealVectorBounds,
    baseUMazeGoalState,
    baseUMazeStateValidityChecker,
    path_to_numpy,
    copySE3State2Data,
    copyData2SE3State,
)

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc


def visualize_path(data: str, goal=[0, 16]):
    """
    From https://ompl.kavrakilab.org/pathVisualization.html
    """
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
    UMaze_x = np.array([-2, 10, 10, -2, -2, 6, 6, -2, -2, -2]) / 4 * 8
    UMaze_y = np.array([-2, -2, 10, 10, 6, 6, 2, 2, 2, -2]) / 4 * 8
    ax.plot(UMaze_x, UMaze_y, "r")

    plt.xlim(-8, 24)
    plt.ylim(-8, 24)
    plt.legend()
    plt.show()


class IRLCostObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super(IRLCostObjective, self).__init__(si)
        self.cost_fn = cost_fn

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        # state = qpos + qvel = 15 + 14 = 29
        s1_temp = np.empty(29, dtype=np.float32)
        s2_temp = np.empty(29, dtype=np.float32)

        copySE3State2Data(state=s1[0], data=s1_temp)
        copySE3State2Data(state=s2[0], data=s2_temp)

        # joint state and velocity state
        for i in range(7, 15):
            s1_temp[i] = s1[1][i - 7]
            s2_temp[i] = s2[1][i - 7]

        for j in range(15, 29):
            s1_temp[j] = s1[2][j - 15]
            s2_temp[j] = s2[2][j - 15]

        c = self.cost_fn(s1_temp, s2_temp)
        # return ob.Cost(c)
        # TODO:
        return ob.Cost(1.0)


def getIRLCostObjective(si, cost_fn) -> ob.OptimizationObjective:
    return IRLCostObjective(si, cost_fn)


class AntUMazeGoalState(baseUMazeGoalState):
    def __init__(self, si: ob.SpaceInformation, goal: np.ndarray, threshold: float):
        super(AntUMazeGoalState, self).__init__(si, goal, threshold)

    def sampleGoal(self, state: ob.State) -> None:

        SE3_goal_data = list(chain(self.goal, [0], [0, 0, 0, 1]))

        copyData2SE3State(data=SE3_goal_data, state=state[0])

        # following are dummy values needed for properly setting the goal state
        # only using x and y to calculate the distance to the goal
        for i in range(7, 15):
            state[1][i - 7] = 0

        for j in range(15, 29):
            state[2][j - 15] = 0


class AntStateValidityChecker(baseUMazeStateValidityChecker):
    def __init__(
        self,
        si: Union[oc.SpaceInformation, ob.SpaceInformation],
        size: float = 0.25 + 2 * 0.08,
        scaling: float = 8,
    ):
        super(AntStateValidityChecker, self).__init__(si, size, scaling)


class AntStatePropagator(oc.StatePropagator):
    def __init__(self, si: oc.SpaceInformation, agent_model: AgentModel):
        super().__init__(si)
        self.si = si
        self.agent_model = agent_model
        import ipdb; ipdb.set_trace()

        # A placeholder for qpos, qvel and control in propagte function that don't waste time on numpy creation
        self.qpos_temp = np.zeros(15)
        self.qvel_temp = np.zeros(14)
        self.action_temp = np.zeros(8)

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # Copy ompl state to qpos and qvel
        copySE3State2Data(state=state[0], data=self.qpos_temp)
        for i in range(7, 15):
            self.qpos_temp[i] = state[1][i - 7]

        for j in range(15, 29):
            self.qvel_temp[j - 15] = state[2][j - 15]

        # copy OMPL contorl to Mujoco (8D)
        for i in range(self.action_temp.shape[0]):
            self.action_temp[i] = control[i]

        # Set qpos and qvel to Mujoco sim state
        self.agent_model.set_state(self.qpos_temp, self.qvel_temp)
        # Simulate in Mujoco
        self.agent_model.do_simulation(
            self.action_temp, n_frames=self.agent_model.frame_skip
        )
        # Obtain new qpos and qvel from Mujoco sim
        next_obs = self.agent_model._get_obs()

        # Copy qpos and qvel to OMPL state
        copyData2SE3State(data=next_obs, state=result[0])
        for i in range(7, 15):
            state[1][i - 7] = next_obs[i]

        for j in range(15, 29):
            state[2][j - 15] = next_obs[j]

    def canPropagateBackward(self) -> bool:
        return False

    def canSteer(self) -> bool:
        return False


class BasePlannerAntUMaze:
    def __init__(self, env, use_control=False, log_level=0):
        """
        other_wrapper(<TimeLimit<MazeEnv<PointUMaze-v0>>>)  --> <MazeEnv<PointUMaze-v0>>
        """
        self.agent_model = env.unwrapped.wrapped_env

        # space information
        # nq = 15 -> R3 + SO3 + 8D
        # [x, y, z,
        # qw, qx, qy, qz,
        # hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4]

        # R3 -> [x, y, z]
        self.R3_high = [20, 20, 1]
        self.R3_low = [-4, -4, 0]

        # q (4 dim) -> [qw, qx, qy, qz]
        self.unit_quaternion_high = [1, 1, 1, 1]
        self.unit_quaternion_low = [-1, -1, -1, -1]

        # 8 Joints

        # self.joints_high = [0.656203,  1.3356,    0.665058,  0.1000,   0.666901,   0.099308,   0.657522,   1.334417]
        # self.joints_low =  [-0.663243, -0.09996, -0.661253, -1.32870, -0.658905	, -1.34018, -0.656940, -0.09989]

        self.joints_high = np.deg2rad([38, 76.5, 38, 5.73, 38, 5.73, 38, 76.5])
        self.joints_low = np.deg2rad([-38, -5.73, -38, -76.5, -38, -76.5, -38, -5.73])

        # self.joints_high = np.deg2rad([30, 70, 30, -30, 30, -30, 30, 70])
        # self.joints_low =  np.deg2rad([-30, 30, -30,-70, -30,-70, -30, 30])

        self.qpos_high = np.concatenate(
            [self.R3_high, self.unit_quaternion_high, self.joints_high]
        )
        self.qpos_low = np.concatenate(
            [self.R3_low, self.unit_quaternion_low, self.joints_low]
        )

        # nv = 14
        # [x_dot, y_dot, z_dot, qx_dot, qy_dot, qz_dot,
        # hip1_dot, ankle1_dot, hip2_dot, ankle2_dot, hip3_dot, ankle3_dot, hip4_dot, ankle4_dot]
        self.qvel_high = np.ones(14) * 10
        self.qvel_low = -self.qvel_high

        # nu = 8
        # [hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3]
        # ? what does u represents? torque?

        # ======================================================================================
        self.state_dim = 15 + 14
        self.control_dim = 8

        # state bound
        self.state_high = np.concatenate([self.qpos_high, self.qvel_high])
        self.state_low = np.concatenate([self.qpos_low, self.qvel_low])

        # control bound
        self.control_high = np.ones(8) * 30
        self.control_low = -self.control_high

        # goal position and goal radius
        self.goal_pos = np.array([0, 16])
        self.threshold = 0.6

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
        # State Space (A compound space include SE3, R8 and accosicated velocity).
        # SE2 = R^2 + SO2. Should not set the bound for SO2 since it is enfored automatically.
        SE3_space = ob.SE3StateSpace()
        R3_bounds = make_RealVectorBounds(
            bounds_dim=3,
            low=self.R3_low,
            high=self.R3_high,
        )
        SE3_space.setBounds(R3_bounds)

        # 8D of joint space
        joint_space = ob.RealVectorStateSpace(8)
        J_bounds = make_RealVectorBounds(
            bounds_dim=8,
            low=self.joints_low,
            high=self.joints_high,
        )
        joint_space.setBounds(J_bounds)

        # velocity space (R3 + SO3 + 8D) -> 14D
        velocity_space = ob.RealVectorStateSpace(14)
        v_bounds = make_RealVectorBounds(
            bounds_dim=14,
            low=self.qvel_low,
            high=self.qvel_high,
        )
        velocity_space.setBounds(v_bounds)

        # Add subspace to the compound space.
        space = ob.CompoundStateSpace()
        space.addSubspace(SE3_space, 1.0)
        space.addSubspace(joint_space, 1.0)
        space.addSubspace(velocity_space, 1.0)

        # Lock this state space. This means no further spaces can be added as components.
        if space.isCompound() and lock:
            space.lock()
        return space

    def makeStartState(self, s0: np.ndarray) -> ob.State:
        """
        Create a start state.
        """
        if isinstance(s0, np.ndarray):
            assert s0.ndim == 1
        start_state = ob.State(self.space)

        # * Quaternion q = w + xi + yj + zk
        # * Mujoco is [w,x,y,z] while OMPL order is [x,y,z,w], so we swap state[3] and state[6]
        state_tmp = s0.copy()
        state_tmp[3], state_tmp[6] = state_tmp[6], state_tmp[3]

        for i in range(len(s0)):
            # * Copy an element of an array to a standard Python scalar
            # * to ensure C++ can recognize it.
            assert (
                self.state_low[i] <= state_tmp[i] <= self.state_high[i]
            ), f"Index {i}: {[self.state_low[i], self.state_high[i]]}: {state_tmp[i]}"

            start_state[i] = state_tmp[i].item()
        return start_state

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
            # raise NotImplementedError("Control space is not implemented yet.")
            self.cspace = self.makeControlSpace(self.space)
            # Define a simple setup class
            self.ss = oc.SimpleSetup(self.cspace)
            # Retrieve current instance of Space Information
            self.si = self.ss.getSpaceInformation()

            # Set State Propagator
            propagator = AntStatePropagator(self.si, self.agent_model)
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
        stateValidityChecker = AntStateValidityChecker(self.si)
        self.ss.setStateValidityChecker(stateValidityChecker)
        # Set the goal state
        goal_state = AntUMazeGoalState(self.si, self.goal_pos, self.threshold)
        self.ss.setGoal(goal_state)

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
                [
                    [u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]]
                    for u in control_path.getControls()
                ],
                dtype=np.float32,
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
            # Return the states and controls(which is None in og plannig)
            return states, controls
        else:
            raise ValueError("OMPL is not able to solve under current cost function")


class ControlPlanner(BasePlannerAntUMaze):
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
        raise NotImplementedError("Control planning is still not working.")
        return super().control_plan(start_state, solveTime)


class GeometricPlanner(BasePlannerAntUMaze):
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


if __name__ == "__main__":
    import gym
    import mujoco_maze

    env = gym.make("AntUMaze-v0")
    # planner = ControlPlanner(env, plannerType="RRT", log_level=2)
    planner = GeometricPlanner(env, plannerType="RRTstar", log_level=2)

    obs = env.reset()
    start_state = np.concatenate([obs[:-1]])

    state, _ = planner.plan(start_state, 5)
    print(state.shape)

    visualize_path(state)
