from typing import Optional, Callable

import numpy as np

from ompl import util as ou
from ompl import base as ob


from irl.utils import planner_utils


class ReacherGoal(ob.GoalState):
    """
    Defines a goal region around goal state with threshold 
    In Reacher-v2, move two joint such that fingertip position is close to target
    """
    def __init__(
        self, 
        si: ob.SpaceInformation,
        target: np.ndarray,
        threshold: Optional[float] = 0.01
    ):
        super().__init__(si)
        self.target = target
        self.th1, self.th2 = planner_utils.compute_angles_from_xy(self.target[0], self.target[1])
        self.setThreshold(threshold)

    def distanceGoal(self, state: ob.State) -> float:
        """Computes the distance from state to target"""
        finger_pos = planner_utils.compute_xy_from_angles(state[0].value, state[1].value)
        return np.linalg.norm(self.target - finger_pos)

    def sampleGoal(self, state: ob.State) -> None:
        state[0].value = self.th1
        state[1].value = self.th2
        state[2][0] = 0.
        state[2][1] = 0.


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
        # 2 joint angles + 2 angular velocities
        self.state_dim = 4
        # 2 joint torque
        self.control_dim = 2
        ou.setLogLevel(ou.LogLevel.LOG_ERROR)
        # ou.setLogLevel(ou.LogLevel.LOG_WARN)

    def get_StateSpace(self) -> ob.StateSpace:
        """
        Create the state space for Reacher-v2
        State includes [theta, theta_dot]
        """
        # Construct [theta, theta_dot] state space
        # SO2 state space enforces angle to be in [-pi, pi]
        joint0_th_space = ob.SO2StateSpace()
        joint1_th_space = ob.SO2StateSpace()
        joint_thdot_space = ob.RealVectorStateSpace(2)
        joint_thdot_space.setBounds(
            planner_utils.make_RealVectorBounds(
                dim=2,
                low=np.array([-100., -100.]),
                high=np.array([100., 100.])
            )
        )

        # Create compound space which allows the composition of state spaces.
        space = ob.CompoundStateSpace()
        subspaces = [
            joint0_th_space, 
            joint1_th_space, 
            joint_thdot_space
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

    def get_Goal(self, si: ob.SpaceInformation, target: np.ndarray) -> ob.Goal:
        return ReacherGoal(si, target)

    def get_StartState(self, start: np.ndarray) -> ob.State:
        assert isinstance(start, np.ndarray) and start.ndim == 1 and len(start) == self.state_dim

        start_state = ob.State(self.space)
        for i in range(len(start)):
            # * Copy an element of an array to a standard Python scalar
            # * to ensure C++ can recognize it.
            start_state[i] = start[i]

        assert self.state_validity_checker.isValid(start_state), (
            f"Start state {start} is not valid"
        )        
        return start_state

    def update_ss_cost(self, cost_fn: Callable, target: np.ndarray):
        # Set up cost function
        # self.objective = planner_utils.ReacherIRLObjective(self.si, cost_fn, target)
        self.objective = planner_utils.ReacherShortestDistanceObjective(self.si, target)
        self.ss.setOptimizationObjective(self.objective)
