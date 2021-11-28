from typing import Union


from ompl import base as ob
from ompl import control as oc

class MinimumTransitionObjective(ob.PathLengthOptimizationObjective):
    """Minimum number of Transitions"""

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super(MinimumTransitionObjective, self).__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        return ob.Cost(1.0)