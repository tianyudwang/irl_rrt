from typing import Optional, Tuple

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from irl.planners.base_planner import PendulumBasePlanner
from irl.utils import planner_utils

class PendulumStatePropagator(oc.StatePropagator):
    """State propagator function for Pendulum-v1 environment"""
    def __init__(
        self,
        si: oc.SpaceInformation,
        state_validity_checker: ob.StateValidityChecker
    ):
        super().__init__(si)

        assert si.getMinControlDuration() == si.getMaxControlDuration()== 1, (
            "SpaceInformation control duration is not set to 1"
        ) 
        self.si = si
        self.state_validity_checker = state_validity_checker

        # Pendulum specifications
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0

    def propagate(self, state, control, duration, result):
        """
        Define the discrete time dynamics. 
        Computes the next state given current state, control, control duration.
        We assume si.MinMaxControlDuration=1
        """
        assert self.state_validity_checker.isValid(state), (
            f"State {state} is not valid before propagate function"
        )
        assert -2 <= control[0] <= 2, (
            f"Control {control} is not valid before propagate function"
        )

        th, thdot, u = state[0].value, state[1][0], control[0]
        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) 
                             + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = self.angle_normalize(th + newthdot * self.dt)

        result[0].value = newth 
        result[1][0] = newthdot

        assert self.state_validity_checker.isValid(result), (
            f"Result {[result[0].value, result[1][0]]} "
            "is not valid after propagate function"
        )
    
    def canPropagateBackward(self) -> bool:
        """Does not allow backward state propagation in time"""
        return False

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

class PendulumControlPlanner(PendulumBasePlanner):
    """
    Parent class for SST and RRT control planners
    Instantiate StateSpace, ControlSpace, SimpleSetup, 
    SpaceInformation, OptimizationObjective, etc
    """
    def __init__(self):
        super().__init__()

        # First set up StateSpace and ControlSpace and SimpleSetup
        self.space = self.get_StateSpace()
        self.cspace = self.get_ControlSpace(self.space)
        self.ss = oc.SimpleSetup(self.cspace)

        # Add StateValidityChecker to SimpleSetup
        self.si = self.ss.getSpaceInformation()
        self.state_validity_checker = self.get_StateValidityChecker(self.si)
        self.ss.setStateValidityChecker(self.state_validity_checker)

        # Add StatePropagator to SimpleSetup
        self.si.setMinMaxControlDuration(1, 1)
        state_propagator = self.get_StatePropagator(self.si, self.state_validity_checker)
        self.ss.setStatePropagator(state_propagator)

        # Add Goal to SimpleSetup
        goal = self.get_Goal(self.si)
        self.ss.setGoal(goal)

        # Define optimization objective
        self.objective = planner_utils.PendulumShortestDistanceObjective(self.si)
        self.ss.setOptimizationObjective(self.objective)

    def get_ControlSpace(self, space: ob.StateSpace) -> oc.ControlSpace:
        """
        Create the control space for Pendulum-v1
        Control is torque
        """
        # Set parameters for state space
        control_dim = 1
        control_low = np.array([-2.])
        control_high = np.array([2])

        #########################################
        cspace = oc.RealVectorControlSpace(space, control_dim)
        c_bounds = planner_utils.make_RealVectorBounds(
            dim=control_dim,
            low=control_low,
            high=control_high
        )
        cspace.setBounds(c_bounds)

        #########################################
        print(f"\nCreated control space {cspace.getName()} with {cspace.getDimension()} dimensions")

        return cspace

    def get_StatePropagator(self, si, state_validity_checker):
        return PendulumStatePropagator(si, state_validity_checker)

    def plan(
        self, 
        start: np.ndarray, 
        solveTime: Optional[float] = 1.0,
        total_solveTime: Optional[float] = 20.0
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls
        """
        # Clear previous planning data and set new start state
        self.ss.clear()
        start = self.get_StartState(start)
        self.ss.setStartState(start)

        status = self.ss.solve(solveTime)
        t = self.ss.getLastPlanComputationTime()

        while planner_utils.PlannerStatus[status.asString()] != 1 and t <= total_solveTime:
            status = self.ss.solve(solveTime)
            t += self.ss.getLastPlanComputationTime() 

        msg = planner_utils.color_status(status)
        objective = self.ss.getProblemDefinition().getOptimizationObjective()

        if bool(status):
            # Retrieve path and controls
            control_path = self.ss.getSolutionPath()
            geometric_path = control_path.asGeometric()
            controls = control_path.getControls()
            print(
                f"{msg}: "
                f"Path length is {geometric_path.length():.2f}, "
                f"cost is {geometric_path.cost(objective).value():.2f}, ",
                f"solve time is {t:.4f}"
            )
            # Convert to numpy arrays
            states = planner_utils.path_to_numpy(geometric_path, dim=2)
            controls = planner_utils.controls_to_numpy(controls, dim=1)
            return planner_utils.PlannerStatus[status.asString()], states, controls
        else:
            print(status.asString())
            raise ValueError("OMPL is not able to solve under current cost function")

class PendulumSSTPlanner(PendulumControlPlanner):
    def __init__(self):
        super().__init__()

        planner = oc.SST(self.si)
        #TODO: check planner selection/pruning radius
        self.ss.setPlanner(planner)

class PendulumRRTPlanner(PendulumControlPlanner):
    def __init__(self):
        super().__init__()

        planner = oc.RRT(self.si)
        #TODO: check planner nearest neighbor
        self.ss.setPlanner(planner)
