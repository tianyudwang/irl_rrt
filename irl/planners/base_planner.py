from typing import Optional, Callable

import numpy as np

from ompl import util as ou
from ompl import base as ob


from irl.utils import planner_utils


class ReacherGoalState(ob.GoalState):
    """
    Defines a goal region around goal state with threshold 
    In Reacher-v2, reaches goal if fingertip position is close to target
    """
    def __init__(
        self, 
        si: ob.SpaceInformation,
        goal: np.ndarray,
        threshold: Optional[float] = 0.02
    ):
        super().__init__(si)
        self.goal = goal
        self.setThreshold(threshold)

    def distanceGoal(self, state: ob.State) -> float:
        """Computes the distance from state to goal"""
        finger_pos = np.array([state[3][0], state[3][1]])
        return np.linalg.norm(self.goal - finger_pos)

    def sampleGoal(self, state: ob.State) -> None:
        state[3][0] = self.goal[0]
        state[3][1] = self.goal[1]


class ReacherStateValidityChecker(ob.StateValidityChecker):
    """
    Checks whether a given state is a valid/feasible state in Reacher-v2
    State is [theta, theta_dot]
    Only need to check bounds since there are no obstacles in state space
    """
    def __init__(self, si: ob.SpaceInformation):
        super().__init__(si)
        self.si = si

    def isValid(self, state: ob.State) -> bool:
        if isinstance(state, ob.CompoundStateInternal):
            return self.si.satisfiesBounds(state)

        # Start state has type ob.ScopedState, need to get internal state
        elif isinstance(state, ob.State):
            return self.si.satisfiesBounds(state())

        else:
            raise ValueError(f"state type {type(state)} not recognized")

class ReacherBasePlanner:
    """
    Initialize StateSpace, StateValidityChecker, and ProblemDefinition
    To be inherited by specific geometric/control planners
    """

    def __init__(self):
        # 2 joint angles + 2 angular velocities + finger xy
        self.state_dim = 6
        # 2 joint torque
        self.control_dim = 2
        ou.setLogLevel(ou.LogLevel.LOG_ERROR)

    def get_StateSpace(self) -> ob.StateSpace:
        """
        Create the state space for Reacher-v2
        State includes [theta, theta_dot, fingertip_xy]
        """
        # Construct [theta, theta_dot] state space
        # SO2 state space enforces angle to be in [-pi, pi]
        joint0_th_space = ob.SO2StateSpace()
        joint1_th_space = ob.SO2StateSpace()
        joint_thdot_space = ob.RealVectorStateSpace(2)
        joint_thdot_space.setBounds(
            planner_utils.make_RealVectorBounds(
                dim=2,
                low=np.array([-15., -15.]),
                high=np.array([15., 15.])
            )
        )

        finger_space = ob.RealVectorStateSpace(2)
        space_bounds = planner_utils.make_RealVectorBounds(
            dim=2,
            low=np.array([-0.21, -0.21]),
            high=np.array([0.21, 0.21])
        )
        finger_space.setBounds(space_bounds)

        # Create compound space which allows the composition of state spaces.
        space = ob.CompoundStateSpace()
        subspaces = [
            joint0_th_space, 
            joint1_th_space, 
            joint_thdot_space, 
            finger_space
        ]
        for subspace in subspaces:
            space.addSubspace(subspace, 1.0)

        # Lock the compound state space        
        space.lock()
        space.sanityChecks()
        assert space.getDimension() == self.state_dim, (
            f"Constructed state space with {space.getDimension()} dimensions, "
            f"not equal to assigned {self.state_dim}"
        )

        return space

    def get_StateValidityChecker(self, si: ob.SpaceInformation) -> ob.StateValidityChecker:
        return ReacherStateValidityChecker(si)

    def get_Goal(self, si: ob.SpaceInformation, goal: np.ndarray) -> ob.Goal:
        return ReacherGoalState(si, goal)

    def get_StartState(self, start: np.ndarray) -> ob.State:
        if isinstance(start, np.ndarray):
            assert start.ndim == 1
        start_state = ob.State(self.space)
        for i in range(4):
            # * Copy an element of an array to a standard Python scalar
            # * to ensure C++ can recognize it.
            start_state[i] = start[i]

        # import ipdb; ipdb.set_trace()
        assert self.state_validity_checker.isValid(start_state), (
            f"Start state {start} is not valid"
        )        
        # import ipdb; ipdb.set_trace()
        return start_state

    def update_ss_cost(self, cost_fn: Callable, goal: np.ndarray):
        # Set up cost function
        costObjective = planner_utils.ReacherIRLObjective(self.si, cost_fn, goal)
        self.ss.setOptimizationObjective(costObjective)

