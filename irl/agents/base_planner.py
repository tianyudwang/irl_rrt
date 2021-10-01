import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

## @cond IGNORE
# Our "collision checker". Our state space does not contain any obstacles. 
# The checker trivially returns true for any state.
class ValidityChecker(ob.StateValidityChecker):
    # Returns whether the given state's position overlaps the
    # circular obstacle
    def isValid(self, state):
        return True

## Defines an optimization objective by computing the cost of motion between 
# two endpoints.
class IRLCostObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super(IRLCostObjective, self).__init__(si)
        self.cost_fn = cost_fn
    
    def motionCost(self, s1, s2):
        s1 = np.array([s1[0], s1[1]])
        s2 = np.array([s2[0], s2[1]])
        c = self.cost_fn(s1, s2)
        return ob.Cost(c)

def getIRLCostObjective(si, cost_fn):
    return IRLCostObjective(si, cost_fn)


class BasePlanner:
    def __init__(self, state_dim, bounds, goal):
        self.state_dim = state_dim
        self.bounds = bounds 
        self.goal = goal
        self.init_simple_setup()

    def init_simple_setup(self):
        """
        Initialize an ompl::geometric::SimpleSetup instance
        without setting the planner
        """
        # Set log to warn/info/debug
        ou.setLogLevel(ou.LOG_WARN)
        # Construct the state space in which we're planning. We're
        # planning in [-bounds[0],bounds[1]]x[-bounds[0],bounds[1]], a subset of R^2.
        space = ob.RealVectorStateSpace(self.state_dim)  
        space.setBounds(self.bounds[0], self.bounds[1])   
        self.space = space

        # Construct a space information instance for this state space
        si = ob.SpaceInformation(space) 
        # Set the object used to check which states in the space are valid
        validityChecker = ValidityChecker(si)
        si.setStateValidityChecker(validityChecker) 
        si.setup()
        self.si = si

        # Simple setup instance that contains the space information
        ss = og.SimpleSetup(si)

        # Set the agent goal state
        goal = ob.State(space)
        goal[0], goal[1] = self.goal[0], self.goal[1]  
        ss.setGoalState(goal)
        self.ss = ss   

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)  

    def plan(self, start_state, solveTime=0.5):
        """
        :param start_state: start location of the planning problem
        :param solveTime: allowed planning budget
        :return:
            path
        """
        # Clear previous planning data, does not affect settings and start/goal
        self.ss.clear()

        # Reset the start state
        start = ob.State(self.space)
        start[0], start[1] = start_state[0].item(), start_state[1].item() 
        self.ss.setStartState(start)

        # solve and get optimal path
        # TODO: current termination condition is a fixed amount of time for planning
        # Change to exactSolnPlannerTerminationCondition when an exact but suboptimal 
        # path is found
        while not self.ss.getProblemDefinition().hasExactSolution():
            solved = self.ss.solve(solveTime)

        if solved:
            path = self.ss.getSolutionPath().printAsMatrix() 
            path = np.fromstring(path, dtype=float, sep='\n').reshape(-1, self.state_dim)
        else:
            raise ValueError("OMPL is not able to solve under current cost function")
            path = None
        return path