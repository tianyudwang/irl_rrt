import argparse
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

    goal
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

    def sampleGoal(self, state: ob.State) -> None:
        state[0][0] = self.goal[0]
        state[0][1] = self.goal[1]
        # following are dummy values needed for properly setting the goal state
        # only using x and y to calculate the distance to the goal
        for i in range(3):
            state[1][i] = 0

class AntMazeStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si,
        size: float,
        scaling: float,
    ):
        super().__init__(si)
        self.si = si
        # radius of agent (consider as a sphere)
        self.size = size
        self.offset = -2
        self.scaling = scaling 
        assert self.scaling == 4

        unitXMin = unitYMin = -0.5
        unitXMax = unitYMax = 2.5

        unitMidBlockXMin = -0.5
        unitMidBlockXMax = 1.5
        unitMidBlockYMin = 0.5
        unitMidBlockYMax = 1.5

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
        velocity_limits: float,
    ):
        super().__init__(si)
        self.si = si
        self.agent_model = agent_model
        self.velocity_limits = velocity_limits

        # A placeholder for qpos and qvel in propagte function that don't waste time on numpy creation
        self.qpos_temp = np.empty(2)
        self.qvel_temp = np.empty(2)
        self.ctrl_temp = np.empty(2)
    
    # TODO: finishi this
    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # * Control [ballx, bally]
        assert self.si.satisfiesBounds(state), "Input state not in bounds"
        
        # ==== Get qpos and qvel from ompl state ====
        # qpos = [x, y]
        for i in range(2):
            self.qpos_temp[i] = state[0][i]
        # qvel = [vx, vy]
        for j in range(2):
            self.qvel_temp[j] = state[1][j]
        # clip_velocity
        np.clip(self.qvel_temp, -self.velocity_limits, self.velocity_limits, out=self.qvel_temp)
        # === Get control from ompl control ===
        for k in range(2):
            self.ctrl_temp[k] = control[k]
        
        # ==== Propagate qpos and qvel with given control===
        # assume MinMaxControlDuration = 1 and frame_skip = 1
        self.agent_model.set_state(self.qpos_temp, self.qvel_temp)
        self.agent_model.do_simulation(self.ctrl_temp, self.agent_model.frame_skip)
        # obtain new simulation result
        next_obs = self.agent_model._get_obs()

        # ==== Copy Mujoco State to OMPL State ====
        for p in range(2):
            result[0][p] = next_obs[p]
        for q in range(2):
            result[1][q] = next_obs[2+q]
        # ==== End of propagate ====

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
        self.control_dim = 9

        # qpos
        self.qpos_low  = np.array([0, 0]) + self.offset
        self.qpos_high = np.array([3, 3]) + self.offset
        # qvel
        self.qvel_high = np.array([5, 5])
        self.qvel_low = -self.qvel_high 
        
        # state bound
        self.state_high = np.concatenate([self.qpos_high, self.qvel_high])
        self.state_low =  np.concatenate([self.qpos_low, self.qvel_low]) 
        
        # control bound
        self.control_high = np.ones(self.control_dim) * 30
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
        # State Space (A compound space include SO3 and accosicated velocity).
        # SE2 = R^2 + SO2. Should not set the bound for SO2 since it is enfored automatically.
        qpos_space_dim = 2
        qpos_space = ob.RealVectorStateSpace(qpos_space_dim)
        qpos_bounds = make_RealVectorBounds(
            bounds_dim=qpos_space_dim,
            low=self.qpos_low,
            high=self.qpos_high,
        )
        qpos_space.setBounds(qpos_bounds)

        # velocity space.
        qvel_space_dim = 2
        qvel_space = ob.RealVectorStateSpace(qvel_space_dim)
        v_bounds = make_RealVectorBounds(
            bounds_dim=qvel_space_dim,
            low=self.qvel_low,
            high=self.qvel_high,
        )
        qvel_space.setBounds(v_bounds)

        # Add subspace to the compound space.
        space = ob.CompoundStateSpace()
        space.addSubspace(qpos_space, 1.0)
        space.addSubspace(qvel_space, 1.0)

        # Lock this state space. This means no further spaces can be added as components.
        if space.isCompound() and lock:
            space.lock()
        return space

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
        
        return AntMazeStateValidityChecker(self.si, size=0.1, scaling=1.0, offset=0.3)

    def makeStatePropagator(self):
        """
        Create a state propagator.
        """
        return AntMazeStatePropagator(self.si, self.agent_model, velocity_limits=5)

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
            controls = np.asarray([[u[0], u[1]] for u in controls_ompl], dtype=np.float32)
        except:
            return None, None
        return states, controls


class ControlPlanner(BasePlannerAntMaze):
    """
    Control planning using oc.planner
    """

    def __init__(self, env, plannerType: str, goal_pos: np.ndarray, threshold: float, log_level: int = 0):
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

    def __init__(self, env, plannerType: str, goal_pos: np.ndarray, threshold: float, log_level: int = 0):
        super(GeometricPlanner, self).__init__(env, goal_pos, threshold, False, log_level)
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
    parser.add_argument("--render", "-r", action="store_true")
    args = parser.parse_args()
    env = gym.make("antmaze-umaze-v2")
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
    if args.render:
        while 1:
            obs, *_ = env.step(env.action_space.sample())
            # ic(obs[:2])
            env.render()
    visualize_path(goal=[0, 8])
