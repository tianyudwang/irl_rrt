from ompl import control as oc

from irl.planners.base_planner import BasePlanner

class SSTPlanner(BasePlanner):
    def __init__(self):
        super(SSTPlanner, self).__init__()
        self.init_planner()

    def init_planner(self):
        # Set planner
        planner = oc.SST(self.si)
        self.ss.setPlanner(planner)