from typing import Optional

import numpy as np

from ompl import util as ou
from ompl import base as ob


from irl.utils import planner_utils


class PendulumGoalState(ob.GoalState):
    """
    Defines a goal region around goal state with threshold 
    In Pendulum-v1, the goal state is [0, 0]
    """
    def __init__(
        self, 
        si: ob.SpaceInformation, 
        goal: Optional[np.ndarray] = np.array([0., 0.]), 
        threshold: Optional[float] = 0.1
    ):
        super().__init__(si)
        assert len(goal.shape) == 1 and goal.shape[0] == 2
        self.goal = goal.tolist()
        self.setThreshold(threshold)

    def distanceGoal(self, state: ob.State) -> float:
        """Computes the distance from state to goal"""
        dx = state[0].value - self.goal[0]
        dy = state[1][0] - self.goal[1]
        return np.linalg.norm([dx, dy])

    def sampleGoal(self, state: ob.State) -> None:
        state[0].value = self.goal[0]
        state[1][0] = self.goal[1]


class PendulumStateValidityChecker(ob.StateValidityChecker):
    """
    Checks whether a given state is a valid/feasible state in Pendulum-v1
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

class PendulumBasePlanner:
    """
    Initialize StateSpace, StateValidityChecker, and ProblemDefinition
    To be inherited by specific geometric/control planners
    """

    def __init__(self):
        ou.setLogLevel(ou.LogLevel.LOG_ERROR)

    def get_StateSpace(self) -> ob.StateSpace:
        """
        Create the state space for Pendulum-v1
        State includes [theta, theta_dot]
        """
        # Construct [theta, theta_dot] state space
        # SO2 state space enforces angle to be in [-pi, pi]
        state_high = np.array([np.pi, 8], dtype=np.float64)
        th_space = ob.SO2StateSpace()
        th_dot_space = ob.RealVectorStateSpace(1)
        th_dot_bounds = ob.RealVectorBounds(1)
        th_dot_bounds.setLow(-state_high[1])
        th_dot_bounds.setHigh(state_high[1])
        th_dot_space.setBounds(th_dot_bounds)

        # Create compound space which allows the composition of state spaces.
        space = ob.CompoundStateSpace()
        space.addSubspace(th_space, 1.0)
        space.addSubspace(th_dot_space, 1.0)
        
        # Lock the compound state space        
        space.lock()
        space.sanityChecks()

        return space

    def get_StateValidityChecker(self, si: ob.SpaceInformation) -> ob.StateValidityChecker:
        return PendulumStateValidityChecker(si)

    def get_Goal(self, si: ob.SpaceInformation) -> ob.Goal:
        return PendulumGoalState(si)

    def get_StartState(self, start: np.ndarray) -> ob.State:
        if isinstance(start, np.ndarray):
            assert start.ndim == 1
        start_state = ob.State(self.space)
        for i in range(len(start)):
            # * Copy an element of an array to a standard Python scalar
            # * to ensure C++ can recognize it.
            start_state[i] = start[i].item()

        assert self.state_validity_checker.isValid(start_state), (
            f"Start state {start} is not valid"
        )        
        return start_state

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = planner_utils.PendulumIRLObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)


#####################################################################
class BasePlanner:
    def __init__(self):
        # Space information
        self.state_dim = 2
        self.state_low = np.array([-np.pi, -8], dtype=np.float64)
        self.state_high = np.array([np.pi, 8], dtype=np.float64)
        self.control_low = -2.0
        self.control_high = 2.0
        self.goal = np.array([0., 0.], dtype=np.float64)

        # Pendulum parameters
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.05
        self.max_angular_velocity = 8.0
        self.max_torque = 2.0

        self.init_simple_setup()

    def construct_spaces(self):
        # Construct [theta, theta_dot] state space
        # SO2 state space enforces angle to be in [-pi, pi]
        th_space = ob.SO2StateSpace()
        th_dot_space = ob.RealVectorStateSpace(1)
        th_dot_bounds = ob.RealVectorBounds(1)
        th_dot_bounds.setLow(self.state_low[1])
        th_dot_bounds.setHigh(self.state_high[1])
        th_dot_space.setBounds(th_dot_bounds)

        # Create compound space which allows the composition of state spaces.
        space = ob.CompoundStateSpace()
        space.addSubspace(th_space, 1.0)
        space.addSubspace(th_dot_space, 1.0)
        # Lock this state space. This means no further spaces can be added as components.
        space.lock()

        # Create a control space
        cspace = oc.RealVectorControlSpace(space, 1)    
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(self.control_low)
        cbounds.setHigh(self.control_high)
        cspace.setBounds(cbounds)

        return space, cspace

    def isStateValid(self, si, state):
        """perform collision checking or check if other constraints are satisfied"""
        return si.satisfiesBounds(state)

    def propagate(self, start, control, duration, state):
        """
        Define the discrete time dynamics. 
        Computes the next state given current state, control, control duration.
        """
        th, th_dot, u = start[0].value, start[1][0], control[0]

        # Assert states are proper
        assert -np.pi <= th <= np.pi, f"State theta is out of bounds: {th}"
        assert -8. <= th_dot <= 8., f"State theta_dot is out of bounds: {th_dot}"
        assert -2. <= u <= 2, f"Control input u is out of bounds: {u}"

        newthdot = th_dot + (3.0 * self.g / (2.0 * self.l) * np.sin(th) 
                             + 3.0 / (self.m * self.l ** 2) * u) * duration
        newthdot = np.clip(newthdot, -self.max_angular_velocity, self.max_angular_velocity)
        newth = th + newthdot * duration

        state[0].value = newth 
        state[1][0] = newthdot

        # Enforce the angle in SO2
        self.space.enforceBounds(state)


    def init_simple_setup(self):
        """
        Initialize an ompl::control::SimpleSetup instance
        """
        # Set log to warn/info/debug
        ou.setLogLevel(ou.LOG_WARN)

        # Define state and control spaces
        self.space, self.cspace = self.construct_spaces()

        # Define a simple setup class
        ss = oc.SimpleSetup(self.cspace)
        self.si = ss.getSpaceInformation()
        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(partial(self.isStateValid, self.si))
        )
        ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        # Set the agent goal state
        goal = ob.State(self.space)
        goal[0], goal[1] = self.goal[0], self.goal[1]  
        ss.setGoalState(goal)
        self.ss = ss 

        # Set propagation step size -> duration of each step
        self.si.setMinMaxControlDuration(1, 1)
        self.si.setPropagationStepSize(self.dt)


    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)  

    def plan(self, start_state, solveTime=0.5):
        # Clear previous planning data, does not affect settings and start/goal
        self.ss.clear()

        # Reset the start state
        start = ob.State(self.space)
        start[0], start[1] = start_state[0].item(), start_state[1].item() 
        self.ss.setStartState(start)

        status = self.ss.solve(solveTime)

        t = self.ss.getLastPlanComputationTime()
        msg = planner_utils.color_status(status)

        objective = self.ss.getProblemDefinition().getOptimizationObjective()

        if bool(status):    
            control_path = self.ss.getSolutionPath()
            geometric_path = control_path.asGeometric()
            controls = control_path.getControls()
            print(
                f"{msg}: "
                f"Path length is {geometric_path.length():.2f}, "
                f"cost is {geometric_path.cost(objective).value():.2f}, ",
                f"solve time is {t:.2f}"
            )

            # Convert to numpy arrays
            states = planner_utils.path_to_numpy(geometric_path, dim=2)
            controls = planner_utils.controls_to_numpy(controls, dim=1)
            return planner_utils.PlannerStatus[status.asString()], states, controls
        else:
            raise ValueError("OMPL is not able to solve under current cost function")
