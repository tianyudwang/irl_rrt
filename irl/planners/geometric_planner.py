from typing import Optional, Tuple

import numpy as np

from ompl import base as ob
from ompl import geometric as og

from irl.planners.base_planner import PendulumBasePlanner
from irl.utils import planner_utils


class PendulumGeometricPlanner(PendulumBasePlanner):
    """
    Parent class for RRT* and PRM* planners
    Instantiate StateSpace, SimpleSetup, SpaceInformation, OptimizationObjective, etc
    """

    def __init__(self):
        super().__init__()

        # First set up StateSpace and ControlSpace and SimpleSetup
        self.space = self.get_StateSpace()
        self.ss = og.SimpleSetup(self.space)

        # Add StateValidityChecker to SimpleSetup
        self.si = self.ss.getSpaceInformation()
        self.state_validity_checker = self.get_StateValidityChecker(self.si)
        self.ss.setStateValidityChecker(self.state_validity_checker)

        # Add Goal to SimpleSetup
        goal = self.get_Goal(self.si)
        self.ss.setGoal(goal)

        # Define optimization objective
        self.objective = planner_utils.PendulumShortestDistanceObjective(self.si)
        self.ss.setOptimizationObjective(self.objective)

    def plan(
        self, 
        start: np.ndarray, 
        solveTime: Optional[float] = 2.0
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls (None for geometric planners)
        """
        # Clear previous planning data and set new start state
        self.ss.clear()
        start = self.get_StartState(start)
        self.ss.setStartState(start)        
        
        status = self.ss.solve(solveTime)
        t = self.ss.getLastPlanComputationTime()
                 
        msg = planner_utils.color_status(status)
        objective = self.ss.getProblemDefinition().getOptimizationObjective()
        if bool(status):
            # Retrieve path
            geometric_path = self.ss.getSolutionPath()
            print(
                f"{msg}: "
                f"Path length is {geometric_path.length():.2f}, "
                f"cost is {geometric_path.cost(objective).value():.2f}, ",
                f"solve time is {t:.2f}"
            )
            states = planner_utils.path_to_numpy(geometric_path, dim=2)
            return planner_utils.PlannerStatus[status.asString()], states, None
        else:
            print(status.asString())
            raise ValueError("OMPL is not able to solve under current cost function")

class PendulumRRTstarPlanner(PendulumGeometricPlanner):
    def __init__(self):
        super().__init__()

        self.planner = og.RRTstar(self.si)
        self.planner.setRange(0.5)        
        self.ss.setPlanner(self.planner)

class PendulumPRMstarPlanner(PendulumGeometricPlanner):
    def __init__(self):
        super().__init__()

        self.planner = og.LazyPRMstar(self.si)
        #TODO: check planner range for PRM and PRMstar
        self.planner.setRange(0.5)
        self.ss.setPlanner(self.planner)