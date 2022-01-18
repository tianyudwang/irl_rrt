from typing import Union, Optional, List

import os
from collections import OrderedDict
from math import pi

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

PlannerStatus = {
    'Exact solution': 1,
    'Approximate solution': 2,
}

class Maze2DIRLObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super().__init__(si)
        self.cost_fn = cost_fn

        self.s1_data = np.empty(4, dtype=np.float32)
        self.s2_data = np.empty(4, dtype=np.float32)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        """Query the neural network cost function for a cost between two states"""
        self.cp_state_to_data(s1, self.s1_data)
        self.cp_state_to_data(s2, self.s2_data)

        c = self.cost_fn(self.s1_data, self.s2_data)
        return ob.Cost(c)

    def cp_state_to_data(self, state: ob.State, data: np.ndarray):
        # 2D position and velocity
        # ob.State is a CompoundState of 2 RealVectorState's        
        data[0] = state[0][0]
        data[1] = state[0][1]
        data[2] = state[1][0]
        data[3] = state[1][1]

class AntMazeIRLObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super().__init__(si)
        self.cost_fn = cost_fn

        self.s1_data = np.empty(29, dtype=np.float32)
        self.s2_data = np.empty(29, dtype=np.float32)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        """Query the neural network cost function for a cost between two states"""
        self.cp_state_to_data(s1, self.s1_data)
        self.cp_state_to_data(s2, self.s2_data)

        c = self.cost_fn(self.s1_data, self.s2_data)
        return ob.Cost(c)

    def cp_state_to_data(self, state: ob.State, data: np.ndarray):
        # ob.State is a CompoundState of an SE3State and 2 RealVectorState's       
        # SE3
        data[0] = state[0].getX()
        data[1] = state[0].getY()
        data[2] = state[0].getZ()
        data[3] = state[0].rotation().x
        data[4] = state[0].rotation().y
        data[5] = state[0].rotation().z
        data[6] = state[0].rotation().w

        # 8 joints
        for i in range(8):
            data[7+i] = state[1][i]
        # 14 velocities
        for i in range(14):
            data[15+i] = state[2][i]


class MinimumTransitionObjective(ob.PathLengthOptimizationObjective):
    """Minimum number of transitions"""

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super().__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        return ob.Cost(1.0)


class Maze2DShortestDistanceObjective(ob.PathLengthOptimizationObjective):
    """
    Shortest path length in 2D
    Assumes state[0] is ob.RealVectorStateInternal
    """

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super().__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        distance = np.linalg.norm([s1[0][0] - s2[0][0], s1[0][1] - s2[0][1]])
        return ob.Cost(distance)

class AntMazeShortestDistanceObjective(ob.PathLengthOptimizationObjective):
    """
    Shortest path length in 2D
    Assumes state[0] is ob.SE3StateInternal
    """

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super().__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        distance = np.linalg.norm([s1[0].getX() - s2[0].getX(), s1[0].getY() - s2[0].getY()])
        return ob.Cost(distance)

def make_RealVectorBounds(
        dim: int, 
        low: np.ndarray, 
        high: np.ndarray
    ) -> ob.RealVectorBounds:
    assert isinstance(dim, int), "dim must be an integer"
    # *OMPL's python binding might not recognize numpy array. convert to list to make it work
    if isinstance(low, np.ndarray):
        assert low.ndim == 1
        low = low.tolist()

    if isinstance(high, np.ndarray):
        assert high.ndim == 1
        high = high.tolist()
    assert isinstance(low, list), "low should be a list or 1D numpy array"
    assert isinstance(high, list), "high should be a list or 1D numpy array"
    assert dim == len(low) == len(high), "Bounds dimensions do not match"

    vector_bounds = ob.RealVectorBounds(dim)
    for i in range(dim):
        vector_bounds.setLow(i, low[i])
        vector_bounds.setHigh(i, high[i])
        # Check if the bounds are valid (same length for low and high, high[i] > low[i])
        vector_bounds.check()
    return vector_bounds


def path_to_numpy(
        path: Union[og.PathGeometric, oc.PathControl],
        dim: int
    ) -> np.ndarray:
    """Convert OMPL path to numpy array"""
    states = np.fromstring(
        path.printAsMatrix(), 
        dtype=np.float32, 
        sep="\n"
    ).reshape(-1, dim)
    return states

def controls_to_numpy(
        controls: List[oc.Control],
        dim: int
    ) -> np.ndarray:
    """Convert OMPL controls to numpy array"""
    controls_np = [np.empty(dim) for _ in range(len(controls))]

    for i in range(len(controls)):
        for j in range(dim):
            controls_np[i][j] = controls[i][j]
    return controls_np


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
    Both OMPL and mujoco quaternions are scalar last [x, y, z, w]
    """
    data[0] = state.x
    data[1] = state.y
    data[2] = state.z
    data[3] = state.w


def copySE3State2Data(state: ob.State, data: np.ndarray) -> None:
    """
    Copy SE3 state to data (modified in place)
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


def visualize_path(
        data: np.ndarray, 
        goal: np.ndarray, 
        save: Optional[bool] = False
    ):
    """From https://ompl.kavrakilab.org/pathVisualization.html"""
    from matplotlib import pyplot as plt

    if data.ndim == 1:
        data = data.reshape(1, -1)

    plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.axes(projection="3d") if not save else plt

    # path
    ax.plot(data[:, 0], data[:, 1], "o-")

    # start
    ax.plot(
        data[0, 0], 
        data[0, 1], 
        "go", 
        markersize=10, 
        markeredgecolor="k", 
        label="start"
    )

    # achieved goal
    ax.plot(
        data[-1, 0],
        data[-1, 1],
        "ro",
        markersize=10,
        markeredgecolor="k",
        label="achieved goal",
    )

    # desired goal
    ax.plot(
        goal[0], 
        goal[1], 
        "bo", 
        markersize=10, 
        markeredgecolor="k", 
        label="desired goal"
    )

    # Grid
    UMaze_x = np.array([0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 0.5, 0.5])
    UMaze_y = np.array([0.5, 0.5, 2.5, 2.5, 0.5, 0.5, 3.5, 3.5, 0.5])
    ax.plot(UMaze_x, UMaze_y, "r")
    plt.legend()
    if save:
        plt.grid()
        plt_dir = "./plots"
        
        if not (os.path.exists(plt_dir)):
            os.makedirs(plt_dir)
        
        import random
        num = random.randint(0, 100)
        plt.savefig(f"{plt_dir}/error_{num}.png")

    else:
        plt.show()
    plt.close()

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

status2color = {
    ob.PlannerStatus.APPROXIMATE_SOLUTION: 'yellow',
    ob.PlannerStatus.EXACT_SOLUTION: 'green',
    ob.PlannerStatus.TIMEOUT: 'red',
}

def color_status(status):
    return colorize(status.asString(), status2color[status.getStatus()])