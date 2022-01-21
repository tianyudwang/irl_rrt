from typing import Union, Optional, List, Callable

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


class PendulumIRLObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn: Callable):
        super().__init__(si)
        self.cost_fn = cost_fn

        self.s1_data = np.empty(2, dtype=np.float32)
        self.s2_data = np.empty(2, dtype=np.float32)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        """Query the neural network cost function for a cost between two states"""
        self.cp_state_to_data(s1, self.s1_data)
        self.cp_state_to_data(s2, self.s2_data)

        c = self.cost_fn(self.s1_data, self.s2_data)
        return ob.Cost(c)

    def cp_state_to_data(self, state: ob.State, data: np.ndarray):
        # theta and theta_dot
        # ob.State is a CompoundState of SO2 and Real       
        data[0] = state[0].value
        data[1] = state[1][0]

class MinimumTransitionObjective(ob.PathLengthOptimizationObjective):
    """Minimum number of transitions"""

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super().__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        return ob.Cost(1.0)

class PendulumShortestDistanceObjective(ob.PathLengthOptimizationObjective):
    """
    Using the cost defined in 
    https://github.com/openai/gym/blob/44242789179d79fae1d5636fa6db23491a5c422e/gym/envs/classic_control/pendulum.py#L39
    OMPL does not allow control cost, thus ignoring the control effort here
    """

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super().__init__(si)

    def stateCost(self, s: ob.State) -> ob.Cost:
        th, thdot = s[0].value, s[1][0]
        assert -np.pi <= th <= np.pi
        c = th ** 2 + 0.1 * thdot ** 2 
        return ob.Cost(c)

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


def path_to_numpy(path: Union[og.PathGeometric, oc.PathControl], dim: int) -> np.ndarray:
    """Convert OMPL path to numpy array"""
    states = np.fromstring(
        path.printAsMatrix(), 
        dtype=np.float32, 
        sep="\n"
    ).reshape(-1, dim)
    return states

def controls_to_numpy(controls: List[oc.Control], dim: int) -> np.ndarray:
    """Convert OMPL controls to numpy array"""
    controls_np = [np.empty(dim) for _ in range(len(controls))]
    for i in range(len(controls)):
        for j in range(dim):
            controls_np[i][j] = controls[i][j]
    return controls_np

def visualize_path(data: np.ndarray):
    assert data.shape[1] == 2

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

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
    goal = [0, 0]
    ax.plot(
        goal[0], 
        goal[1], 
        "bo", 
        markersize=10, 
        markeredgecolor="k", 
        label="desired goal"
    )

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-8, 8)
    plt.legend()
    plt.show()

#######################################################################

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