import argparse
import random
import warnings

from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict

import numpy as np
from mujoco_py import MjSim

import irl.mujoco_ompl_py.mujoco_ompl_interface as mj_ompl

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

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

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


def init_planning(sim: MjSim, param: Dict[str, Any]):
    # Construct the State Space we are planning in [theta, theta_dot]

    si = mj_ompl.createSpaceInformation(
        m=sim.model,
        include_velocity=param["include_velocity"],
    )
    space = si.getStateSpace()
    if space.isCompound():
        printSubspaceInfo(space, param["start"], param["include_velocity"])

    # Define a simple setup class
    ss = oc.SimpleSetup(si)

    # Set state validation check
    mj_validityChecker = mj_ompl.MujocoStateValidityChecker(
        si, sim, include_velocity=param["include_velocity"]
    )
    ss.setStateValidityChecker(mj_validityChecker)

    # Set State Propagator
    mj_propagator = mj_ompl.MujocoStatePropagator(
        si, sim, include_velocity=param["include_velocity"]
    )
    ss.setStatePropagator(mj_propagator)

    # Set propagation step size
    si.setPropagationStepSize(sim.model.opt.timestep)

    # Create a start state and a goal state
    start_state = ob.State(si)
    goal_state = ob.State(si)

    for i in range(param["start"].shape[0]):
        start_state[i] = param["start"][i]
        goal_state[i] = param["goal"][i]

    # Set the start state and goal state
    ss.setStartAndGoalStates(start_state, goal_state, 0.05)

    # Allocate and set the planner to the SimpleSetup
    planner = allocateControlPlanner(si, plannerType=param["plannerType"])
    ss.setPlanner(planner)

    # Set optimization objective
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    return ss


def plan(
    ss: ob.SpaceInformation, runtime: float, state_dim: int
) -> Tuple[oc.PathControl, og.PathGeometric, np.ndarray]:
    """Attempt to solve the problem"""
    assert isinstance(state_dim, int)

    solved = ss.solve(runtime)
    controlPath = None
    controlPath_np = None
    geometricPath = None
    geometricPath_np = None
    if solved:
        # Print the path to screen
        controlPath = ss.getSolutionPath()
        controlPath.interpolate()

        geometricPath = controlPath.asGeometric()
        # geometricPath.interpolate()

        geometricPath_np = np.fromstring(
            geometricPath.printAsMatrix(), dtype=float, sep="\n"
        ).reshape(
            -1,
        )
        # print("Found solution:\n%s" % path)
    else:
        print("No solution found")
    return controlPath, geometricPath, geometricPath_np


def angle_normalize(x) -> float:
    """Converts angles outside of +/-PI to +/-PI"""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def make_RealVectorBounds(bounds_dim: int, low, high) -> ob.RealVectorBounds:
    assert isinstance(bounds_dim, int), "bonds_dim must be an integer"
    # *OMPL's python binding might not recognize numpy array. convert to list to make it work
    if isinstance(low, np.ndarray):
        assert (
            low.ndim == 1
        ), "low should be a 1D numpy array to prevent order mismatch when convert to list"
        low = low.tolist()

    if isinstance(high, np.ndarray):
        assert (
            high.ndim == 1
        ), "high should be a 1D numpy array to prevent order mismatch when convert to list"

        high = high.tolist()
    assert isinstance(low, list), "low should be a list"
    assert isinstance(high, list), "high should be a list"
    assert (
        len(low) == len(high) == bounds_dim
    ), "low and high must have same length as bonds_dim"

    vector_bounds = ob.RealVectorBounds(bounds_dim)
    for i in range(bounds_dim):
        if low[i] == high[i]:
            warnings.warn(
                "\n Although it's OK to set a dummy RealVectorBounds with low == high, "
                + "OMPL planning needs lower bound must be stricly less than upper bound "
                + "Please specify them manually!"
            )
        vector_bounds.setLow(i, low[i])
        vector_bounds.setHigh(i, high[i])
        # Check if the bounds are valid (same length for low and high, high[i] > low[i])
        vector_bounds.check()
    return vector_bounds


def printBounds(bounds: ob.RealVectorBounds, title: str) -> None:
    assert isinstance(bounds, ob.RealVectorBounds)
    print(f"{title}:")
    for i, (low, high) in enumerate(zip(bounds.low, bounds.high)):
        print(f"  Bound {i}: {[low, high]}")
    print()


def printSubspaceInfo(
    space: ob.CompoundStateSpace,
    start: Optional[np.ndarray] = None,
    include_velocity: bool = False,
) -> dict:
    space_dict = OrderedDict()
    print("\nSubspace info: ")
    last_subspace_idx = 0
    for i in range(space.getSubspaceCount()):
        subspace = space.getSubspace(i)
        name = subspace.getName()
        space_dict[name] = subspace
        if isinstance(subspace, ob.RealVectorStateSpace):
            low, high = subspace.getBounds().low, subspace.getBounds().high

        elif isinstance(subspace, ob.SO2StateSpace):
            low, high = [[-np.pi], [np.pi]]

        elif isinstance(subspace, ob.SE2StateSpace):
            low, high = subspace.getBounds().low, subspace.getBounds().high
            # SO2 bound is not inlude in bounds manually add it for visualization
            low.append(-np.pi)
            high.append(np.pi)
        if include_velocity and i == space.getSubspaceCount() / 2:
            print("\n  Velocy State Space:")

        for j in range(len(low)):
            print(f"  {i}: {name}[{j}]\t[{low[j]}, {high[j]}]")
            if start is not None:
                assert low[j] <= start[i + j] <= high[j], (
                    f"start value: {start[i+j]} "
                    + f"is not in range [{low[j]}, {high[j]}] "
                    + f"at subspace ({i}) with inner index ({j})."
                )
            last_subspace_idx += 1

    return space_dict


def CLI():
    parser = argparse.ArgumentParser(description="OMPL Control planning")
    parser.add_argument(
        "--env_id",
        "-env",
        type=str,
        help="Envriment to interact with",
        choices=["InvertedPendulum-v2", "PointUMaze-v0", "AntUMaze-v0"],
        required=True,
    )
    parser.add_argument(
        "-t",
        "--runtime",
        type=float,
        default=5.0,
        help="(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        choices=["RRT", "SST", "KPIECE", "KPIECE1"],
        default="RRT",
        help="The planner to use, either RRT or SST or KPIECE",
    )
    parser.add_argument(
        "-i",
        "--info",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG. Defaults to WARN.",
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--plot", "-p", help="Render environment", action="store_true")
    parser.add_argument(
        "--render", "-r", help="Render environment", action="store_true"
    )
    parser.add_argument(
        "--visual", "-v", help="visulaize environment", action="store_true"
    )
    parser.add_argument(
        "--dummy_setup",
        "-d",
        help="a naive setup for State Space and Control Space relied solely on xml file",
        action="store_true",
    )
    parser.add_argument(
        "--verbose", help="Print additional information", action="store_true"
    )
    parser.add_argument("--render_video", "-rv", help="Save a gif", action="store_true")
    args = parser.parse_args()
    return args
