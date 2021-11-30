from itertools import chain
from typing import Tuple, Union


import numpy as np
from mujoco_maze.agent_model import AgentModel


from irl.agents.base_planner_UMaze import (
    BasePlannerUMaze,
    baseUMazeGoalState,
    baseUMazeStateValidityChecker,
)
from irl.agents.planner_utils import (
    allocateControlPlanner,
    allocateGeometricPlanner,
    make_RealVectorBounds,
    copySE3State2Data,
    copyData2SE3State,
)

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc


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


class BasePlannerAntUMaze(BasePlannerUMaze):
    def __init__(self, env, use_control=False, log_level=0):
        """
        other_wrapper(<TimeLimit<MazeEnv<PointUMaze-v0>>>)  --> <MazeEnv<PointUMaze-v0>>
        """
        super().__init__()
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
        self.control_high = np.ones(8)  # * 30
        self.control_low = -self.control_high

        # goal position and goal radius
        self.goal_pos = np.array([0, 16])
        self.threshold = 0.6
        self.scale = 8

        self.space: ob.StateSpace = None
        self.cspace: oc.ControlSpace = None
        self.init_simple_setup(use_control, log_level)

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
        # * Quaternion q = w + xi + yj + zk
        # * Mujoco is [w,x,y,z] while OMPL order is [x,y,z,w], so we swap state[3] and state[6]
        state_tmp = s0.copy()
        state_tmp[3], state_tmp[6] = state_tmp[6], state_tmp[3]
        return super().makeStartState(state_tmp)

    def makeGoalState(self):
        """
        Create a goal state
        """
        return AntUMazeGoalState(self.si, self.goal_pos, self.threshold)

    def makeStateValidityChecker(self):
        """
        Create a state validity checker.
        """
        return AntStateValidityChecker(self.si)

    def makeStatePropagator(self):
        """
        Create a state propagator.
        """
        return AntStatePropagator(self.si, self.agent_model)

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)

    def control_plan(
        self, start_state: np.ndarray, solveTime: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform control planning for a specified amount of time.
        """
        states, controls_ompl = super().control_plan(start_state, solveTime)
        controls = np.asarray(
            [[u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]] for u in controls_ompl],
            dtype=np.float32,
        )
        return states, controls


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
        # raise NotImplementedError("Control planning is still not working.")
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
    from planner_utils import visualize_path

    env = gym.make("AntUMaze-v0")
    # planner = ControlPlanner(env, plannerType="RRT", log_level=2)
    planner = GeometricPlanner(env, plannerType="RRTstar", log_level=2)

    obs = env.reset()
    start_state = np.concatenate([obs[:-1]])

    state, _ = planner.plan(start_state, 5)
    print(state.shape)

    visualize_path(state, goal=[0, 16], scale=8)
