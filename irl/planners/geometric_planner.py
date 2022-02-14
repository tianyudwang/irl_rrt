from typing import Optional, Tuple

import numpy as np

from ompl import base as ob
from ompl import geometric as og

from irl.planners.base_planner import ReacherBasePlanner
from irl.utils import planner_utils


class ReacherGeometricPlanner(ReacherBasePlanner):
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

        # Define optimization objective
        self.objective = planner_utils.MinimumTransitionObjective(self.si)
        self.ss.setOptimizationObjective(self.objective)

    def plan(
        self, 
        start: np.ndarray, 
        goal: np.ndarray,
        solveTime: Optional[float] = 1.0,
        total_solveTime: Optional[float] = 10.0
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls (None for geometric planners)
        """
        # Clear previous planning data and set new start state
        self.ss.clear()
        self.ss.setStartState(self.get_StartState(start))        
        self.ss.setGoal(self.get_Goal(self.si, goal))

        status = self.ss.solve(solveTime)
        t = self.ss.getLastPlanComputationTime()

        while not self.ss.haveExactSolutionPath() and t <= total_solveTime:
            status = self.ss.solve(solveTime)
            t += self.ss.getLastPlanComputationTime()
                 
        msg = planner_utils.color_status(status)
        objective = self.ss.getProblemDefinition().getOptimizationObjective()
        if bool(status):
            # Retrieve path
            geometric_path = self.ss.getSolutionPath()
            # geometric_path.interpolate()
            states = geometric_path.getStates()
            print(
                f"{msg}: "
                f"Path length is {len(states)}, "
                f"cost is {geometric_path.cost(objective).value():.2f}, ",
                f"solve time is {t:.2f}"
            )
            states = planner_utils.states_to_numpy(states)
            return status.asString(), states, None
        else:
            print(status.asString())
            raise ValueError("OMPL is not able to solve under current cost function")

class ReacherRRTstarPlanner(ReacherGeometricPlanner):
    def __init__(self):
        super().__init__()

        self.planner = og.RRTstar(self.si)
        self.planner.setRange(2.0)        
        self.ss.setPlanner(self.planner)

class ReacherPRMstarPlanner(ReacherGeometricPlanner):
    def __init__(self):
        super().__init__()

        self.planner = og.PRMstar(self.si)
        #TODO: check planner range for PRM and PRMstar
        # self.planner.setRange(1.0)
        self.ss.setPlanner(self.planner)

    def plan(
        self, 
        start: np.ndarray, 
        goal: np.ndarray,
        solveTime: Optional[float] = 1.0,
        total_solveTime: Optional[float] = 10.0
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls (None for geometric planners)
        """
        # Clear previous query and set new start and goal states
        # self.ss.clearQuery()
        # self.ss.getPlanner().clearQuery()
        # self.ss.clear()
        self.planner.clearQuery()
        self.ss.setStartState(self.get_StartState(start))        
        self.ss.setGoal(self.get_Goal(self.si, goal))

        status = self.ss.solve(solveTime)
        t = self.ss.getLastPlanComputationTime()

        while not self.ss.haveExactSolutionPath() and t <= total_solveTime:
            status = self.ss.solve(solveTime)
            t += self.ss.getLastPlanComputationTime()
                 
        msg = planner_utils.color_status(status)
        objective = self.ss.getProblemDefinition().getOptimizationObjective()
        if bool(status):
            # Retrieve path
            geometric_path = self.ss.getSolutionPath()
            # geometric_path.interpolate()
            states = geometric_path.getStates()
            print(
                f"{msg}: "
                f"Path length is {geometric_path.length():.2f}, "
                f"cost is {geometric_path.cost(objective).value():.2f}, ",
                f"solve time is {t:.2f}"
            )
            states = planner_utils.states_to_numpy(states)
            return status.asString(), states, None
        else:
            print(status.asString())
            raise ValueError("OMPL is not able to solve under current cost function")