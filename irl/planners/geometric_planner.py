from typing import Optional, Tuple

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from irl.planners.base_planner import Maze2DBasePlanner, AntMazeBasePlanner
import irl.planners.planner_utils as planner_utils

class Maze2DGeometricPlanner(Maze2DBasePlanner):
    """
    Parent class for RRT* and PRM* planners
    Instantiate StateSpace, SimpleSetup, SpaceInformation, OptimizationObjective, etc
    """

    def __init__(self, timeLimit: Optional[float] = None):
        super().__init__(timeLimit)

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
        self.objective = planner_utils.Maze2DShortestDistanceObjective(self.si)
        self.ss.setOptimizationObjective(self.objective)


    # def plan(
    #     self, 
    #     start: np.ndarray, 
    #     solveTime: Optional[float] = 2.0,
    #     clear: Optional[bool] = True
    # ) -> Tuple[int, np.ndarray, np.ndarray]:
    #     """
    #     Return a list of states and controls (None for geometric planners)
    #     """
    #     # Clear previous planning data and set new start state
    #     if clear:
    #         self.ss.clear()
    #     start = self.get_StartState(start)
    #     self.ss.setStartState(start)        
        
    #     status = self.ss.solve(solveTime)
    #     t = self.ss.getLastPlanComputationTime()
        
    #     if self.timeLimit is not None:
    #         while not self.ss.haveExactSolutionPath() and t < self.timeLimit:
    #             print(f"\t{t:.1f}/{self.timeLimit:.1f}", end="\r") 
    #             status = self.ss.solve(1.0)
    #             t += self.ss.getLastPlanComputationTime()
                 
    #     msg = planner_utils.color_status(status)
    #     if bool(status):
    #         # Retrieve path
    #         geometricPath = self.ss.getSolutionPath()
    #         print(
    #             f"{msg}: "
    #             f"Path length is {geometricPath.length():.2f}, "
    #             f"cost is {geometricPath.cost(self.objective).value():.2f}, ",
    #             f"solve time is {t:.2f}"
    #         )
    #         states = planner_utils.path_to_numpy(geometricPath, dim=4)
    #         return planner_utils.PlannerStatus[status.asString()], states, None
    #     else:
    #         print(status.asString())
    #         raise ValueError("OMPL is not able to solve under current cost function")

    def plan_exact_solution(
        self, 
        start: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Plan until an exact solution is found"""
        # Clear previous planning data and set new start state
        self.ss.clear()
        start = self.get_StartState(start)
        self.ss.setStartState(start) 

        termination_condition = ob.PlannerStatus.EXACT_SOLUTION 

        status = self.ss.solve(termination_condition)
        t = self.ss.getLastPlanComputationTime()

        msg = planner_utils.color_status(status)
        if bool(status):
            # Retrieve path
            geometricPath = self.ss.getSolutionPath()
            print(
                f"{msg}: "
                f"Path length is {geometricPath.length():.2f}, "
                f"cost is {geometricPath.cost(self.objective).value():.2f}, ",
                f"solve time is {t:.2f}"
            )
            states = planner_utils.path_to_numpy(geometricPath, dim=4)
            return planner_utils.PlannerStatus[status.asString()], states, None
        else:
            print(status.asString())
            raise ValueError("OMPL is not able to solve under current cost function")




class Maze2DRRTstarPlanner(Maze2DGeometricPlanner):
    def __init__(self, timeLimit: Optional[float] = None):
        super().__init__(timeLimit)

        self.planner = og.RRTstar(self.si)
        self.planner.setRange(1.0)        # Maximum range of a motion to be added to tree
        self.ss.setPlanner(self.planner)

class Maze2DPRMstarPlanner(Maze2DGeometricPlanner):
    def __init__(self, timeLimit: Optional[float] = None):
        super().__init__(timeLimit)

        self.planner = og.LazyPRMstar(self.si)
        #TODO: check planner range for PRM and PRMstar
        self.planner.setRange(1.0)
        self.ss.setPlanner(self.planner)

    # def plan_exact_solution(
    #     self, 
    #     start: np.ndarray,
    #     clear_query: Optional[bool] = False
    # ) -> Tuple[int, np.ndarray, np.ndarray]:
    #     """Plan until an exact solution is found"""
    #     # Clear previous queries and set new start state
    #     if clear_query:
    #         self.planner.clearQuery()
    #     else:
    #         self.ss.clear()
    #     start = self.get_StartState(start)
    #     self.ss.setStartState(start) 

    #     termination_condition = ob.PlannerStatus.EXACT_SOLUTION 

    #     status = self.ss.solve(termination_condition)
    #     t = self.ss.getLastPlanComputationTime()

    #     msg = planner_utils.color_status(status)
    #     if bool(status):
    #         # Retrieve path
    #         geometricPath = self.ss.getSolutionPath()
    #         print(
    #             f"{msg}: "
    #             f"Path length is {geometricPath.length():.2f}, "
    #             f"cost is {geometricPath.cost(self.objective).value():.2f}, ",
    #             f"solve time is {t:.2f}"
    #         )
    #         states = planner_utils.path_to_numpy(geometricPath, dim=4)
    #         return planner_utils.PlannerStatus[status.asString()], states, None
    #     else:
    #         print(status.asString())
    #         raise ValueError("OMPL is not able to solve under current cost function")


#####################################################################

class AntMazeGeometricPlanner(AntMazeBasePlanner):
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
        objective = planner_utils.AntMazeShortestDistanceObjective(self.si)
        self.ss.setOptimizationObjective(objective)

    def plan(
        self, 
        start: np.ndarray, 
        solveTime: Optional[float] = 15.0
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls (None for geometric planners)
        """
        # Clear previous planning data and set new start state
        self.ss.clear()
        start = self.get_StartState(start)
        self.ss.setStartState(start)

        status = self.ss.solve(solveTime)
        print(status.asString())
        if bool(status):
            # Retrieve path
            geometricPath = self.ss.getSolutionPath()
            states = planner_utils.path_to_numpy(geometricPath, dim=29)
            return planner_utils.PlannerStatus[status.asString()], states, None
        else:
            raise ValueError("OMPL is not able to solve under current cost function")

class AntMazeRRTstarPlanner(AntMazeGeometricPlanner):
    def __init__(self):
        super().__init__()

        planner = og.RRTstar(self.si)
        planner.setRange(10.0)        # Maximum range of a motion to be added to tree
        self.ss.setPlanner(planner)

class AntMazePRMstarPlanner(AntMazeGeometricPlanner):
    def __init__(self):
        super().__init__()

        planner = og.LazyPRMstar(self.si)
        #TODO: check planner range for PRM and PRMstar
        planner.setRange(10.0)
        self.ss.setPlanner(planner)

