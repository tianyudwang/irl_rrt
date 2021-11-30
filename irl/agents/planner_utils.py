from collections import OrderedDict
from math import pi
from typing import Union, Optional

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc


def allocateGeometricPlanner(si: ob.SpaceInformation, plannerType: str) -> ob.Planner:
    """Allocate planner in OMPL Geometric"""
    # Keep these in alphabetical order and all lower case
    if plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    else:
        ou.OMPL_ERROR(f"Planner-type {plannerType} is not implemented.")


def allocateControlPlanner(si: ob.SpaceInformation, plannerType: str) -> ob.Planner:
    """Allocate planner in OMPL Control"""
    # Keep these in alphabetical order and all lower case
    if plannerType.lower() == "rrt":
        return oc.RRT(si)
    elif plannerType.lower() == "sst":
        return oc.SST(si)
    else:
        ou.OMPL_ERROR(f"Planner-type {plannerType} is not implemented.")


def angle_normalize(x: float) -> float:
    return ((x + pi) % (2 * pi)) - pi


def make_RealVectorBounds(bounds_dim: int, low, high) -> ob.RealVectorBounds:
    assert isinstance(bounds_dim, int), "bonds_dim must be an integer"
    # *OMPL's python binding might not recognize numpy array. convert to list to make it work
    if isinstance(low, np.ndarray):
        assert low.ndim == 1
        low = low.tolist()

    if isinstance(high, np.ndarray):
        assert high.ndim == 1
        high = high.tolist()
    assert isinstance(low, list), "low should be a list or 1D numpy array"
    assert isinstance(high, list), "high should be a list or 1D numpy array"

    vector_bounds = ob.RealVectorBounds(bounds_dim)
    for i in range(bounds_dim):
        vector_bounds.setLow(i, low[i])
        vector_bounds.setHigh(i, high[i])
        # Check if the bounds are valid (same length for low and high, high[i] > low[i])
        vector_bounds.check()
    return vector_bounds


def path_to_numpy(
    path: Union[og.PathGeometric, oc.PathControl], state_dim: int, dtype: np.dtype
) -> np.ndarray:
    """Convert OMPL path to numpy array"""
    assert isinstance(state_dim, int)
    return np.fromstring(path.printAsMatrix(), dtype=dtype, sep="\n").reshape(
        -1, state_dim
    )


def printSubspaceInfo(
    space: ob.CompoundStateSpace,
    start: Optional[np.ndarray] = None,
    include_velocity: bool = False,
) -> dict:
    space_dict = OrderedDict()
    print("\nSubspace info: ")
    last_subspace_idx = 0
    k = 0
    for i in range(space.getSubspaceCount()):
        subspace = space.getSubspace(i)
        name = subspace.getName()
        space_dict[name] = subspace
        if isinstance(subspace, ob.RealVectorStateSpace):
            low, high = subspace.getBounds().low, subspace.getBounds().high

        elif isinstance(subspace, ob.SO2StateSpace):
            low, high = [[-np.pi], [np.pi]]

        elif isinstance(subspace, ob.SO3StateSpace):
            low, high = [None] * 4, [None] * 4

        elif isinstance(subspace, ob.SE2StateSpace):
            low, high = subspace.getBounds().low, subspace.getBounds().high
            # SO2 bound is not inluded in bounds manually add it for visualization
            low.append(-np.pi)
            high.append(np.pi)

        elif isinstance(subspace, ob.SE3StateSpace):
            low, high = subspace.getBounds().low, subspace.getBounds().high

        for j in range(len(low)):
            print(f"  {k}|{i}: {name}[{j}]\t[{low[j]}, {high[j]}]")
            if start is not None:
                assert low[j] <= start[i + j] <= high[j], (
                    f"start value: {start[i+j]} "
                    + f"is not in range [{low[j]}, {high[j]}] "
                    + f"at subspace ({i}) with inner index ({j})."
                )
            last_subspace_idx += 1
            k += 1

    return space_dict


def copyR3State2Data(state: ob.State, data: np.ndarray) -> None:
    """
    Copy R3 state to data (modified in place)
    """
    assert isinstance(data, np.ndarray)
    data[0] = state.getX()
    data[1] = state.getY()
    data[2] = state.getZ()


def copySO3State2Data(state: ob.State, data: np.ndarray) -> None:
    """
    Copy SO3 state to data (modified in place)
    OMPL SO3 is [x, y, z, w]
    Mujoco is [w, x, y, z]
    """
    data[0] = state.w
    data[1] = state.x
    data[2] = state.y
    data[3] = state.z


def copySE3State2Data(state: ob.State, data: np.ndarray) -> None:
    """
    Copy SE3 state to data (modified in place)
    OMPL SE3 is [x, y, z, qx, qy, qz, qw,]
    Mujoco is [x, y, z, qw, qx, qy, qz,]
    """
    copyR3State2Data(state, data[0:3])
    copySO3State2Data(state.rotation(), data[3:7])


def copySE2State2Data(state: ob.State, data: np.ndarray) -> None:
    """
    Copy SE2 state to data (modified in place)
    """
    data[0] = state.getX()
    data[1] = state.getY()
    data[2] = state.getYaw()


def copyData2SE2State(
    data: np.ndarray,
    state: ob.State,
) -> None:
    """
    Copy SE2 state to data (modified in place)
    """
    state.setX(data[0])
    state.setY(data[1])
    state.setYaw(data[2])


def copyData2SE3State(data: np.ndarray, state: ob.State) -> None:
    """
    Copy data to SE3 state (modified in place)
    Mujoco is [x, y, z, qw, qx, qy, qz,]
    OMPL SE3 is [x, y, z, qx, qy, qz, qw,]
    """
    state.setXYZ(data[0], data[1], data[2])
    state.rotation().w = data[3]
    state.rotation().x = data[4]
    state.rotation().y = data[5]
    state.rotation().z = data[6]


def visualize_path(data: np.ndarray, goal: np.ndarray, scale: float, save:  bool  = False):
    """
    From https://ompl.kavrakilab.org/pathVisualization.html
    """
    from matplotlib import pyplot as plt
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # ax = plt.axes(projection="3d")
    ax = plt

    plt.figure(figsize=(10, 10), dpi=300)

    # path
    ax.plot(data[:, 0], data[:, 1], "o-")

    circle1 = plt.Circle((data[0, 0], data[0, 1]), 0.5, color='g', lw=5, label="start")
    plt.gca().add_patch(circle1)
    
    # ax.plot(
        # data[0, 0], data[0, 1], "go", markersize=10, markeredgecolor="k", 
    # )
    ax.plot(
        data[-1, 0],
        data[-1, 1],
        "ro",
        markersize=10,
        markeredgecolor="k",
        label="achieved goal",
    )
    ax.plot(
        goal[0], goal[1], "bo", markersize=10, markeredgecolor="k", label="desired goal"
    )

    # Grid
    UMaze_x = np.array([-0.5, 2.5, 2.5, -0.5, -0.5, 1.5, 1.5, -0.5, -0.5, -0.5]) * scale
    UMaze_y = np.array([-0.5, -0.5, 2.5, 2.5, 1.5, 1.5, 0.5, 0.5, 0.5, -0.5]) * scale
    ax.plot(UMaze_x, UMaze_y, "r")

    plt.legend()
    if save:
        plt.grid()
        plt.savefig("./plots/error.png")
    else:
        plt.show()
    plt.close()