from typing import Optional, Tuple

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from irl.planners.base_planner import Maze2DBasePlanner
import irl.planners.planner_utils as planner_utils

class Maze2DGeometricPlanner(Maze2DBasePlanner):
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
        objective = planner_utils.ShortestDistanceObjective(self.si)
        self.ss.setOptimizationObjective(objective)


    def plan(
        self, 
        start: np.ndarray, 
        solveTime: Optional[float] = 5.0
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls (None for geometric planners)
        """
        # Clear previous planning data and set new start state
        self.ss.clear()
        start = self.get_StartState(start)
        self.ss.setStartState(start)

        status = self.ss.solve(solveTime)
        if bool(status):
            # Retrieve path
            geometricPath = self.ss.getSolutionPath()
            states = planner_utils.path_to_numpy(geometricPath)
            return planner_utils.PlannerStatus[status.asString()], states, None
        else:
            raise ValueError("OMPL is not able to solve under current cost function")

class Maze2DRRTstarPlanner(Maze2DGeometricPlanner):
    def __init__(self):
        super().__init__()

        planner = og.RRTstar(self.si)
        planner.setRange(1.0)        # Maximum range of a motion to be added to tree
        self.ss.setPlanner(planner)

class Maze2DPRMstarPlanner(Maze2DGeometricPlanner):
    def __init__(self):
        super().__init__()

        planner = og.LazyPRMstar(self.si)
        #TODO: check planner range for PRM and PRMstar
        planner.setRange(1.0)
        self.ss.setPlanner(planner)




