import random

import numpy as np

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(
        0, join(dirname(dirname(dirname(abspath(__file__)))), "ompl/py-bindings")
    )
    from ompl import util as ou
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og


def setLogLevel(level: int) -> None:
    """Set the log leve"""
    if level == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif level == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif level == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")


def setRandomSeed(seed: int) -> None:
    """Set the random seed"""
    ou.RNG(seed)
    random.seed(seed)
    np.random.seed(seed)


def allocateGeometricPlanner(si: ob.SpaceInformation, plannerType: str) -> ob.Planner:
    """Allocate planner in OMPL Geometric"""
    # Keep these in alphabetical order and all lower case
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in og allocation function.")


def allocateControlPlanner(si: ob.SpaceInformation, plannerType: str) -> ob.Planner:
    """Allocate planner in OMPL Control"""
    # Keep these in alphabetical order and all lower case
    if plannerType.lower() == "kpiece" or plannerType.lower() == "kpiece1":
        return oc.KPIECE1(si)
    elif plannerType.lower() == "rrt":
        return oc.RRT(si)
    elif plannerType.lower() == "sst":
        return oc.SST(si)
    else:
        ou.OMPL_ERROR(
            f"Planner-type {plannerType} is not implemented in oc allocation function."
        )


def getPathLengthObjective(si: ob.SpaceInformation):
    return ob.PathLengthOptimizationObjective(si)


def getThresholdPathLengthObj(si: ob.SpaceInformation):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj


class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si: ob.SpaceInformation):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to maximize path clearance from obstacles,
    # but we want to represent the objective as a path cost
    # minimization. Therefore, we set each state's cost to be the
    # reciprocal of its clearance, so that as state clearance
    # increases, the state cost decreases.
    def stateCost(self, s: ob.State):
        return ob.Cost(
            1 / (self.si_.getStateValidityChecker().clearance(s) + sys.float_info.min)
        )


def getClearanceObjective(si: ob.SpaceInformation):
    return ClearanceObjective(si)


def getBalancedObjective1(si: ob.SpaceInformation):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 5.0)
    opt.addObjective(clearObj, 1.0)
    return opt


def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj


def angle_normalize(x) -> float:
    """Converts angles outside of +/-PI to +/-PI"""
    return ((x + np.pi) % (2 * np.pi)) - np.pi
