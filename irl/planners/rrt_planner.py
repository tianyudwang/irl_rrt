from ompl import control as oc

from irl.planners.base_planner import BasePlanner

class RRTPlanner(BasePlanner):
    def __init__(self):
        super(RRTPlanner, self).__init__()
        self.init_planner()

    def init_planner(self):
        """
        Initialize an ompl::geometric::SimpleSetup instance
        Check out https://ompl.kavrakilab.org/genericPlanning.html
        """
        # Set up RRT planner
        planner = oc.RRT(self.si)
        # Set the maximum length of a motion
        planner.setRange(0.1)
        self.ss.setPlanner(planner)
