from typing import Tuple, Optional, Union

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

import irl.planners.planner_utils as planner_utils


class Maze2DGoalState(ob.GoalState):
    """
    Defines a goal region around goal state with threshold 
    In umaze, the goal region is defined as the states whose distance to goal 
    is smaller than threshold=0.1 in x-y plane
    """
    def __init__(
        self, 
        si: ob.SpaceInformation, 
        goal: Optional[np.ndarray] = np.array([1., 1.]), 
        threshold: Optional[float] = 0.1
    ):
        super().__init__(si)
        assert len(goal.shape) == 1 and goal.shape[0] == 2
        self.goal = goal.tolist()
        self.setThreshold(threshold)

    def distanceGoal(self, state: ob.State) -> float:
        """Computes the distance from state to goal"""
        dx = state[0][0] - self.goal[0]
        dy = state[0][1] - self.goal[1]
        return np.linalg.norm([dx, dy])

    def sampleGoal(self, state: ob.State) -> None:
        state[0][0] = self.goal[0]
        state[0][1] = self.goal[1]

        state[1][0], state[1][1] = 0., 0.

class Maze2DStateValidityChecker(ob.StateValidityChecker):
    """
    Checks whether a given state is a valid/feasible state in umaze

    The maze2d-umaze-v1 string spec is 
    U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"
    with start at (3, 1) and goal at (1, 1)
    The empty region is the square [0.5, 3.5] x [0.5, 3.5], 
    excluding the rectangle [1.5, 2.5] x [0.5, 2.5]
    The point size is 0.1, thus the feasible area for point center is 
    [0.6, 3.4] x [0.6, 3.4], excluding [1.4, 2.6] x [0.6, 2.6]
    """
    def __init__(
        self,
        si,
        size: Optional[float] = 0.1
    ):
        super().__init__(si)
        self.si = si
        self.size = size        # radius of point

        # Square extents
        self.square_x_min = 0.5 + self.size
        self.square_x_max = 3.5 - self.size
        self.square_y_min = 0.5 + self.size
        self.square_y_max = 3.5 - self.size

        # Rectangle extents
        self.rect_x_min = 1.5 - self.size
        self.rect_x_max = 2.5 + self.size
        self.rect_y_min = 0.5 + self.size
        self.rect_y_max = 2.5 + self.size

    def isValid(self, state: ob.State) -> bool:
        """
        State is valid if x-y position is inside square but outside rectangle
        """
        if isinstance(state, ob.CompoundStateInternal):
            x, y = state[0][0], state[0][1]
            if not self.si.satisfiesBounds(state):
                return False
        elif isinstance(state, ob.State):
            x, y = state[0], state[1]

        # print(state, type(state))
        # assert isinstance(state, ob.CompoundStateInternal), (
        #     f"state type is {type(state)}"
        # )
        # x, y = state[0][0], state[0][1]
        # x, y = state[0], state[1]
        # print(state, type(state))

        in_square = all([
            self.square_x_min < x < self.square_x_max,
            self.square_y_min < y < self.square_y_max
        ])

        in_rect = all([
            self.rect_x_min <= x <= self.rect_x_max,
            self.rect_y_min <= y <= self.rect_y_max
        ])

        return in_square and not in_rect


class Maze2DBasePlanner:
    """
    Initialize StateSpace, StateValidityChecker, and ProblemDefinition
    To be inherited by specific geometric/control planners
    """
    def __init__(self):
        pass

    def get_StateSpace(self) -> ob.StateSpace:
        """
        Create the state space for maze2d-umaze-v1
        State includes qpos (x, y) and qvel (vx, vy)
        """
        # Set parameters for state space
        qpos_space_dim = 2
        qpos_low = np.array([0.5, 0.5])
        qpos_high = np.array([3.5, 3.5])

        qvel_space_dim = 2
        qvel_low = np.array([-5., -5.])
        qvel_high = np.array([5., 5.])

        #######################################

        qpos_space = ob.RealVectorStateSpace(qpos_space_dim)
        qpos_bounds = planner_utils.make_RealVectorBounds(
            dim=qpos_space_dim,
            low=qpos_low,
            high=qpos_high
        )
        qpos_space.setBounds(qpos_bounds)

        qvel_space = ob.RealVectorStateSpace(qvel_space_dim)
        qvel_bounds = planner_utils.make_RealVectorBounds(
            dim=qvel_space_dim,
            low=qvel_low,
            high=qvel_high,
        )
        qvel_space.setBounds(qvel_bounds)

        space = ob.CompoundStateSpace()
        space.addSubspace(qpos_space, 1.0)
        space.addSubspace(qvel_space, 1.0)

        # Lock the compound state space
        space.lock()            
        space.sanityChecks()

        #########################################
        print(f"\nCreated state space {space.getName()} with {space.getDimension()} dimensions")
        bounds = [subspace.getBounds() for subspace in space.getSubspaces()]
        lows, highs = [], []
        for bound in bounds:
            lows.extend([bound.low[i] for i in range(len(bound.low))])
            highs.extend([bound.high[i] for i in range(len(bound.high))]) 
        print(f"State space lower bound is {lows}")
        print(f"State space upper bound is {highs}")
        return space

    def get_StateValidityChecker(
            self, 
            si: ob.SpaceInformation
        ) -> ob.StateValidityChecker:
        return Maze2DStateValidityChecker(si)

    def get_Goal(self, si: ob.SpaceInformation) -> ob.Goal:
        return Maze2DGoalState(si)

    def get_StartState(self, start: np.ndarray) -> ob.State:
        if isinstance(start, np.ndarray):
            assert start.ndim == 1
        start_state = ob.State(self.space)
        for i in range(len(start)):
            # * Copy an element of an array to a standard Python scalar
            # * to ensure C++ can recognize it.
            start_state[i] = start[i].item()

        # Use the internal compound state
        # import pdb; pdb.set_trace()
        # start_state = start_state()
        assert self.state_validity_checker.isValid(start_state), (
            f"Start state {start} is not  valid"
        )        

        return start_state

