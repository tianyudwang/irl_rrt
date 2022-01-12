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




def visualize_path(data, goal, save=False):
    """"""
    fig = plt.figure()
    offset = 0.3
    size = 0.1
    
    # path
    plt.plot(data[:, 0], data[:, 1], "x-")

    plt.plot(
        data[0, 0],
        data[0, 1],
        "go",
        markersize=10,
        markeredgecolor="k", 
        label="start"
    )
    plt.plot(
        data[-1, 0],
        data[-1, 1],
        "ro",
        markersize=10,
        markeredgecolor="k",
        label="achieved goal",
    )
    plt.plot(
        goal[0],
        goal[1],
        "bo",
        markersize=10,
        markeredgecolor="k",
        label="desired goal"
    )

    # UMaze boundary
    UMaze_x = np.array([0., 1., 1., 2., 2., 3., 3., 0., 0.]) + offset
    UMaze_y = np.array([0., 0., 2., 2., 0., 0., 3., 3., 0.]) + offset
    plt.plot(UMaze_x, UMaze_y, "r")
    
    
    # feasible region
    # UMaze_feasible_x = np.array([0.1, 0.9, 0.9, 2.1, 2.1, 2.9, 2.9, 0.1, 0.1]) + offset
    UMaze_feasible_x = UMaze_x.copy()
    UMaze_feasible_x[0] += size
    UMaze_feasible_x[1:3] -= size
    UMaze_feasible_x[3:5] += size
    UMaze_feasible_x[5:7] -= size
    UMaze_feasible_x[7:] += size
    
    # UMaze_feasible_y = np.array([0.1, 0.1, 2.1, 2.1, 0.1, 0.1, 2.9, 2.9, 0.1]) + offset
    UMaze_feasible_y = UMaze_y.copy()
    UMaze_feasible_y[:6] += size
    UMaze_feasible_y[6:8] -= size
    UMaze_feasible_y[8] += size
    plt.plot(UMaze_feasible_x, UMaze_feasible_y, "k--")
    
    # achived goal with radius
    achieved_circle = plt.Circle(
        xy=(data[-1, 0], data[-1, 1]),
        radius=0.1,
        color="r",
        lw=1,
        label="achieved region"
    )
    plt.gca().add_patch(achieved_circle)
    
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
    
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "path.png"))
    else:
        plt.show()


class PointUMazeGoalState(baseUMazeGoalState):
    def __init__(self, si: ob.SpaceInformation, goal: np.ndarray, threshold: float):
        super(PointUMazeGoalState, self).__init__(si, goal, threshold)
        # goal = [0.8, 0.8]
        # threshold = 0.5

    def sampleGoal(self, state: ob.State) -> None:
        state[0][0] = self.goal[0]
        state[0][1] = self.goal[1]
        # following are dummy values needed for properly setting the goal state
        # only using x and y to calculate the distance to the goal
        for i in range(3):
            state[1][i] = 0
        # ? DO WE NEED TO USE ALL STATES AS GOAL STATE?
        # In d4rl, it is not a first exit problem.
        # The point is required to stay in the goal region(<=0.5) to obatin higher reward. 

class PointStateValidityChecker(baseUMazeStateValidityChecker):
    def __init__(
        self,
        si: Union[oc.SpaceInformation, ob.SpaceInformation],
        size: float,
        scaling: float,
        offset: float,
    ):
        # size is the radius of the robot
        super(PointStateValidityChecker, self).__init__(si, size, scaling, offset)
        
        # size: float = 0.1,
        # scaling: float = 1.0,
        # offset: float = 0.3,


class PointStatePropagator(oc.StatePropagator):
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


class BasePlannerPointUMaze(BasePlannerUMaze):
    def __init__(self, env, goal_pos, goal_threshold, use_control=False, log_level=0):
        """
        other_wrapper(<TimeLimit<MazeEnv<PointUMaze-v0>>>)  --> <MazeEnv<PointUMaze-v0>>
        """
        super().__init__()
        
        # Agent Model
        self.agent_model = env.unwrapped
        
        # UMaze configuraiton
        self.offset = 0.3
        self.scale = 1
        
        # goal
        self._goal_pos = goal_pos
        self._goal_threshold = goal_threshold
                
        # space information
        self.state_dim = 4  # * This is 4 now qpos + qvel (2 each)
        self.control_dim = 2

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
        self.control_high = np.array([1, 1])
        self.control_low = -self.control_high

        self.space: ob.StateSpace = None
        self.cspace: oc.ControlSpace = None
        self.init_simple_setup(use_control, log_level)
        
    @property
    def goal_pos(self):
        return self._goal_pos
    
    @goal_pos.setter
    def goal(self, value):
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
        return PointUMazeGoalState(self.si, self.goal_pos, self.goal_threshold)

    def makeStateValidityChecker(self):
        """
        Create a state validity checker.
        """
        
        return PointStateValidityChecker(self.si, size=0.1, scaling=1.0, offset=0.3)

    def makeStatePropagator(self):
        """
        Create a state propagator.
        """
        return PointStatePropagator(self.si, self.agent_model, velocity_limits=5)

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


class ControlPlanner(BasePlannerPointUMaze):
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


class GeometricPlanner(BasePlannerPointUMaze):
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
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="maze2d-umaze-v1")
    p.add_argument("--plannerType", "-pl", type=str, default="rrtstar")
 
    args = p.parse_args()
    assert args.plannerType.lower() in ["rrt", "sst", "rrtstar", "prmstar"]

    print(f"\nStart testing with {args.plannerType}...")
    env = gym.make(args.env_id)
    print("max_ep_len:", env.spec.max_episode_steps)
    
    use_control = False
    goal_pos = np.array([0.8, 0.8])
    goal_threshold = 0.1
    if args.plannerType.lower() in ["rrt", "sst"]:
        use_control = True
        planner = ControlPlanner(env, args.plannerType, goal_pos, goal_threshold, log_level=2)
    elif args.plannerType.lower() in ["rrtstar", "prmstar"]:
        planner = GeometricPlanner(env, args.plannerType, goal_pos, goal_threshold, log_level=2)
    else:
        raise ValueError("Unknown planner type.")
    
    for i in range(5):
        obs = env.reset()
        old_sim_state = env.unwrapped.sim.get_state()
        
        path, ompl_controls = planner.plan(start_state=obs, solveTime=5)
        # visualize_path(data=path, goal=[0.8, 0.8], save=True)
        visualize_path(data=path, goal=goal_pos, save=True)
        
        if use_control:
            # Ensure we have the same start position
            env.unwrapped.sim.set_state(old_sim_state)
            controls = np.asarray([[u[0], u[1]] for u in ompl_controls])
            ic(controls.shape)
            for u in controls:
                # ic(obs)
                obs, rew, done, info = env.step(u)
                env.render(mode="human")
                time.sleep(0.02)
                if done:
                    ic(info)
                    break
