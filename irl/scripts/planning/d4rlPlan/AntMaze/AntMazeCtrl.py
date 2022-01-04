import argparse
from itertools import chain

import os

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import time
from typing import Union, Tuple

import sys
import random

import gym
import d4rl

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from irl.scripts.planning.d4rlPlan.base_planner_UMaze import (
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

from irl.agents.minimum_transition_objective import MinimumTransitionObjective


def visualize_path(data=None, goal=[0, 16], save=False):
    """ """
    fig = plt.figure()
    offset = -2
    size = 0.25 + 2 * 0.08
    scaling = 4
    # path
    if data is not None:
        plt.plot(data[:, 0], data[:, 1], "o-")
        plt.plot(
            data[0, 0],
            data[0, 1],
            "go",
            markersize=10,
            markeredgecolor="k",
            label="start",
        )
        plt.plot(
            data[-1, 0],
            data[-1, 1],
            "ro",
            markersize=10,
            markeredgecolor="k",
            label="achieved goal",
        )
        # achived goal with radius
        achieved_circle = plt.Circle(
            xy=(data[-1, 0], data[-1, 1]),
            radius=0.1,
            color="r",
            lw=1,
            label="achieved region",
        )
        plt.gca().add_patch(achieved_circle)

    # goal pos
    plt.plot(
        goal[0], goal[1], "bo", markersize=10, markeredgecolor="k", label="desired goal"
    )
    # goal region
    goal_region = plt.Circle(
        xy=(goal[0], goal[1]),
        radius=0.5,
        alpha=0.5,
        color="darkorange",
        lw=1,
        label="goal region",
    )
    plt.gca().add_patch(goal_region)

    # UMaze boundary
    UMaze_x = np.array([0, 3, 3, 0, 0, 2, 2, 0, 0]) * scaling + offset
    UMaze_y = np.array([0, 0, 3, 3, 2, 2, 1, 1, 0]) * scaling + offset
    plt.plot(UMaze_x, UMaze_y, "r")

    # feasible region
    UMaze_feasible_x = UMaze_x + size * np.array([1, -1, -1, 1, 1, 1, 1, 1, 1])
    UMaze_feasible_y = UMaze_y + size * np.array([1, 1, -1, -1, 1, 1, -1, -1, 1])
    plt.plot(UMaze_feasible_x, UMaze_feasible_y, "k--")

    plt.legend()
    plt.grid()
    if save:
        plt.savefig(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "path.png")
        )
    else:
        plt.show()


class AntMazeGoalState(baseUMazeGoalState):
    def __init__(self, si: ob.SpaceInformation, goal: np.ndarray, threshold: float):
        super().__init__(si, goal, threshold)
        # *goal = random sampled
        # threshold = 0.5

    def distanceGoal(self, state: ob.State) -> float:
        """
        Compute the distance to the goal.
        """
        return np.linalg.norm(
            [state[0].getX() - self.goal[0], state[0].getY() - self.goal[1]]
        )

    def sampleGoal(self, state: ob.State) -> None:
        SE3_goal_data = list(chain(self.goal, [0], [0, 0, 0, 1]))
        copyData2SE3State(data=SE3_goal_data, state=state[0])
        # following are dummy values needed for properly setting the goal state
        # only using x and y to calculate the distance to the goal
        for i in range(7, 15):
            state[1][i - 7] = 0

        for j in range(15, 29):
            state[2][j - 15] = 0


class AntMazeStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si,
        size: float,
        scaling: float,
        offset: float,
    ):
        super().__init__(si)
        self.si = si
        # radius of agent (consider as a sphere)
        self.size = size
        self.offset = offset
        self.scaling = scaling
        assert self.size == 0.41, f"{self.size}"
        assert self.scaling == 4
        assert self.offset == -2

        unit_offset = self.offset / self.scaling

        # calculate: base + offset/ scaling

        unitXMin = unitYMin = 0 + unit_offset
        unitXMax = unitYMax = 3 + unit_offset

        unitMidBlockXMin = 0 + unit_offset
        unitMidBlockXMax = 2 + unit_offset
        unitMidBlockYMin = 1 + unit_offset
        unitMidBlockYMax = 2 + unit_offset

        self.Umaze_x_min = self.Umaze_y_min = unitXMin * self.scaling + self.size
        self.Umaze_x_max = self.Umaze_y_max = unitXMax * self.scaling - self.size

        self.midBlock_x_min = unitMidBlockXMin * self.scaling
        self.midBlock_x_max = unitMidBlockXMax * self.scaling + self.size

        self.midBlock_y_min = unitMidBlockYMin * self.scaling - self.size
        self.midBlock_y_max = unitMidBlockYMax * self.scaling + self.size

    def isValid(self, state: ob.State) -> bool:

        # Check if the state is in bound first. If not, return False
        if not self.si.satisfiesBounds(state):
            return False

        x_pos = state[0].getX()
        y_pos = state[0].getY()

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
                    self.midBlock_x_min <= x_pos <= self.midBlock_x_max,
                    self.midBlock_y_min <= y_pos <= self.midBlock_y_max,
                ]
            )
            if inMidBlock:
                valid = False
            else:
                valid = True
        # Not in big square
        else:
            valid = False

        # Inside empty cell
        return valid


class AntMazeStatePropagator(oc.StatePropagator):
    def __init__(
        self,
        si: oc.SpaceInformation,
        agent_model,
    ):
        super().__init__(si)
        self.si = si
        self.agent_model = agent_model

        # A placeholder for qpos and qvel in propagte function that don't waste time on numpy creation
        self.qpos_temp = np.empty(15)
        self.qvel_temp = np.empty(14)
        self.ctrl_temp = np.empty(8)

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # * Control [ballx, bally]
        assert self.si.satisfiesBounds(state), "Input state not in bounds"

        # Copy ompl state to qpos and qvel
        copySE3State2Data(state=state[0], data=self.qpos_temp)
        for i in range(7, 15):
            self.qpos_temp[i] = state[1][i - 7]

        for j in range(15, 29):
            self.qvel_temp[j - 15] = state[2][j - 15]

        # copy OMPL contorl to Mujoco (8D)
        for i in range(self.ctrl_temp.shape[0]):
            self.ctrl_temp[i] = control[i]

        # Set qpos and qvel to Mujoco sim state
        self.agent_model.set_state(self.qpos_temp, self.qvel_temp)
        # Simulate in Mujoco
        self.agent_model.do_simulation(
            self.ctrl_temp, n_frames=self.agent_model.frame_skip
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


class BasePlannerAntMaze(BasePlannerUMaze):
    def __init__(self, env, goal_pos, goal_threshold, use_control=False, log_level=0):
        """
        other_wrapper(<TimeLimit<MazeEnv<PointUMaze-v0>>>)  --> <MazeEnv<PointUMaze-v0>>
        """
        super().__init__()

        # Agent Model
        self.agent_model = env.unwrapped.wrapped_env

        # UMaze configuraiton
        self.offset = -2
        self.scale = 4

        # goal
        self._goal_pos = goal_pos
        self._goal_threshold = goal_threshold

        # space information
        self.state_dim = 29  # * This is 15 qpos + 14 qvel (2 each)
        self.control_dim = 8

        # qpos
        # R3 -> [x,y ,z]
        self.R3_high = np.array([10, 10, 2])
        self.R3_low = np.array([-2, -2, 0])

        # quat(4 dim) -> [qw, qx, qy, qz]
        self.unit_quaternion_high = np.array([1, 1, 1, 1])
        self.unit_quaternion_low = np.array([-1, -1, -1, -1])

        # 8 Joints
        self.joints_high = np.deg2rad([38, 76.5, 38, 5.73, 38, 5.73, 38, 76.5])
        self.joints_low = np.deg2rad([-38, -5.73, -38, -76.5, -38, -76.5, -38, -5.73])

        # qpos: 15 dim
        self.qpos_high = np.concatenate(
            [self.R3_high, self.unit_quaternion_high, self.joints_high]
        )
        self.qpos_low = np.concatenate(
            [self.R3_low, self.unit_quaternion_low, self.joints_low]
        )

        # qvel: 14 dim
        self.qvel_high = np.ones(14) * 10  # ? I cannot find the scaling
        self.qvel_low = -self.qvel_high

        # state bound
        self.state_high = np.concatenate([self.qpos_high, self.qvel_high])
        self.state_low = np.concatenate([self.qpos_low, self.qvel_low])

        # control bound: 8 dim
        self.control_high = np.ones(self.control_dim)  # following the action spcae
        self.control_low = -self.control_high

        self.space: ob.StateSpace = None
        self.cspace: oc.ControlSpace = None
        self.init_simple_setup(use_control, log_level)

    @property
    def goal_pos(self):
        return self._goal_pos

    @goal_pos.setter
    def goal_pos(self, value):
        self._goal_pos = value

    @property
    def goal_threshold(self):
        return self._goal_threshold

    @goal_threshold.setter
    def goal_threshold(self, value):
        self._goal_threshold = value

    def set_goal(self, goal_pos: np.ndarray, threshold: float):
        # goal position and goal radius
        # self.goal_pos = np.array([0.5, 0.5]) + self.offset
        # TODO:
        # ? The threshold is very large.
        self.goal_pos = goal_pos
        self.goal_threshold = threshold

    def makeStateSpace(self, lock: bool = True) -> ob.StateSpace:
        """
        Create a state space.
        """
        # State Space (A compound space include SE3, R8 and accosicated velocity).
        # SE2 = R^2 + SO2. Should not set the bound for SO2 since it is enfored automatically.
        SE3_space = ob.SE3StateSpace()
        R3_bounds = make_RealVectorBounds(
            bounds_dim=self.R3_high.shape[0],
            low=self.R3_low,
            high=self.R3_high,
        )
        SE3_space.setBounds(R3_bounds)

        # 8D of joint space
        joint_space = ob.RealVectorStateSpace(8)
        J_bounds = make_RealVectorBounds(
            bounds_dim=self.joints_high.shape[0],
            low=self.joints_low,
            high=self.joints_high,
        )
        joint_space.setBounds(J_bounds)

        # velocity space (R3 + SO3 + 8D) -> 14D
        velocity_space = ob.RealVectorStateSpace(14)
        v_bounds = make_RealVectorBounds(
            bounds_dim=self.qvel_high.shape[0],
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
        if self._goal_pos is None:
            raise ValueError("Goal position is not set. please call set_goal() first")
        if self._goal_threshold is None:
            raise ValueError("Goal threshold is not set. please call set_goal() first")
        return AntMazeGoalState(self.si, self.goal_pos, self.goal_threshold)

    def makeStateValidityChecker(self):
        """
        Create a state validity checker.
        """

        return AntMazeStateValidityChecker(self.si, size=0.41, scaling=4, offset=-2)

    def makeStatePropagator(self):
        """
        Create a state propagator.
        """
        return AntMazeStatePropagator(self.si, self.agent_model)

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        # costObjective = getIRLCostObjective(self.si, cost_fn)
        # self.ss.setOptimizationObjective(costObjective)
        pass

    def control_plan(
        self, start_state: np.ndarray, solveTime: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform control planning for a specified amount of time.
        """
        states, controls_ompl = super().control_plan(start_state, solveTime)
        # Need to wrap for different number of control dimension
        try:
            controls = np.asarray(
                [
                    [u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]]
                    for u in controls_ompl
                ],
                dtype=np.float32,
            )
        except:
            return None, None
        return states, controls


class ControlPlanner(BasePlannerAntMaze):
    """
    Control planning using oc.planner
    """

    def __init__(
        self,
        env,
        plannerType: str,
        goal_pos: np.ndarray,
        threshold: float,
        log_level: int = 0,
    ):
        super(ControlPlanner, self).__init__(env, goal_pos, threshold, True, log_level)
        self.init_planner(plannerType)

    def init_planner(self, plannerType: str):
        # Set planner
        planner = allocateControlPlanner(self.si, plannerType)
        self.ss.setPlanner(planner)

    def plan(
        self, start_state: np.ndarray, solveTime: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().control_plan(start_state, solveTime)


class GeometricPlanner(BasePlannerAntMaze):
    """
    Geometric planning using og.planner
    """

    def __init__(
        self,
        env,
        plannerType: str,
        goal_pos: np.ndarray,
        threshold: float,
        log_level: int = 0,
    ):
        super(GeometricPlanner, self).__init__(
            env, goal_pos, threshold, False, log_level
        )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--plannerType", "-pl", type=str, default="rrtstar")
    parser.add_argument("--render", "-r", action="store_true")
    args = parser.parse_args()

    env = gym.make("antmaze-umaze-v2")
    env.set_target([1, 9])
    obs = env.reset()
    ic(env.observation_space)
    ic(env.action_space)
    ic(env.spec.max_episode_steps)

    antMazeEnv = env.unwrapped.wrapped_env
    ic(antMazeEnv.init_qpos)
    ic(antMazeEnv.init_qvel)
    ic(antMazeEnv._maze_size_scaling)
    ic(antMazeEnv._maze_height)
    ic(antMazeEnv._np_maze_map)
    ic(antMazeEnv.target_goal)
    ic(antMazeEnv.frame_skip)

    if args.render:
        while 1:
            obs, *_ = env.step(env.action_space.sample())
            # ic(obs[:2])
            env.render()

    goal_pos = np.array([1, 9])
    goal_threshold = 0.5

    if args.plannerType.lower() in ["rrt", "sst"]:
        use_control = True
        planner = ControlPlanner(
            env, args.plannerType, goal_pos, goal_threshold, log_level=2
        )
    elif args.plannerType.lower() in ["rrtstar", "prmstar"]:
        planner = GeometricPlanner(
            env, args.plannerType, goal_pos, goal_threshold, log_level=2
        )

    data, _ = planner.plan(obs, 50)

    visualize_path(data, goal_pos)
