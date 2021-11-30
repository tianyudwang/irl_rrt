from typing import Tuple

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

from irl.agents.planner_utils import make_RealVectorBounds, path_to_numpy, visualize_path


class baseUMazeGoalState(ob.GoalState):
    """
    ompl::base::GoalState (inherits from ompl::base::GoalSampleableRegion) stores one state as the goal.
    Sampling the goal state will always return this state
    and the distance to the goal is implemented by calling ompl::base::StateSpace::distance()
    between the stored goal state and the state passed to ompl::base::GoalRegion::distanceGoal().
    """

    def __init__(self, si: ob.SpaceInformation, goal: np.ndarray, threshold: float):
        super().__init__(si)

        self.si = si
        self.goal = goal[:2].flatten().tolist()
        self.threshold = threshold

        # Set goal threshold
        self.setThreshold(self.threshold)
        assert self.getThreshold() == self.threshold

    def distanceGoal(self, state: ob.State) -> float:
        """
        Compute the distance to the goal.
        """
        return np.linalg.norm(
            [state[0].getX() - self.goal[0], state[0].getY() - self.goal[1]]
        )

    def sampleGoal(self, state: ob.State) -> None:
        raise NotImplementedError("Need to specified goal state according to Env")


class baseUMazeStateValidityChecker(ob.StateValidityChecker):
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

        unitXMin = unitYMin = -0.5
        unitXMax = unitYMax = 2.5

        unitMidBlockXMin = -0.5
        unitMidBlockXMax = 1.5
        unitMidBlockYMin = 0.5
        unitMidBlockYMax = 1.5

        self.scaling = scaling  # 8.0
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


class BasePlannerUMaze:
    @property
    def PropagStepSize(self) -> float:
        return self.agent_model.sim.model.opt.timestep

    def makeStateSpace(self) -> ob.StateSpace:
        """
        Create a state space.
        """
        raise NotImplementedError()

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

            assert self.state_low[i] <= s0[i] <= self.state_high[i], f"Out of Bound at Index {i}: {[self.state_low[i], self.state_high[i]]}: {s0[i]}"

            # * Copy an element of an array to a standard Python scalar
            # * to ensure C++ can recognize it.
            start_state[i] = s0[i].item()
        return start_state

    def makeGoalState(self) -> ob.GoalState:
        """
        Create a goal state.
        """
        raise NotImplementedError()

    def makeStateValidityChecker(self) -> ob.StateValidityChecker:
        """
        Create a state validity checker.
        """
        raise NotImplementedError()

    def makeStatePropagator(self) -> oc.StatePropagator:
        """
        Create a state propagator.
        """
        raise NotImplementedError()

    def update_ss_cost(self):
        raise NotImplementedError()

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
            propagator = self.makeStatePropagator()
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
        stateValidityChecker = self.makeStateValidityChecker()
        self.ss.setStateValidityChecker(stateValidityChecker)
        # Set the goal state
        goal_state = self.makeGoalState()
        self.ss.setGoal(goal_state)

    def clearDataAndSetStartState(self, s0: np.ndarray) -> None:
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
        visualize_path(start_state, self.goal_pos, scale=self.scale, save=True)

        solved = self.ss.solve(solveTime)
        if solved:
            control_path = self.ss.getSolutionPath()
            geometricPath = control_path.asGeometric()
            states_np = path_to_numpy(
                geometricPath, state_dim=self.state_dim, dtype=np.float32
            )
            return states_np, control_path.getControls()
        else:
            visualize_path(start_state, self.goal_pos, scale=self.scale, save=True)
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
