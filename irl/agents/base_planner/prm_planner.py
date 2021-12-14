import numpy as np

from ompl import geometric as og
from irl.agents.base_planner.base_planner import BasePlanner

class PRMPlanner(BasePlanner):
    def __init__(self, state_dim, bounds, goal):
        super(PRMPlanner, self).__init__(state_dim, bounds, goal)
        self.init_planner()

    def init_planner(self):
        """
        Initialize an ompl::geometric::SimpleSetup instance
        Check out https://ompl.kavrakilab.org/genericPlanning.html
        """
        # Set up PRM* planner
        planner = og.PRMstar(self.si)
#        # Set the maximum length of a motion
#        planner.setRange(0.1)
        self.ss.setPlanner(planner)
