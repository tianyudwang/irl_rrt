from typing import Union, Optional, List, Callable, Tuple

import os
from collections import OrderedDict
from math import pi

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

import gym
import torch as th
# import torch.multiprocessing as mp 
import irl.planners.geometric_planner as gp
import irl.planners.control_planner as cp
import irl.utils.pytorch_util as ptu
from irl.utils.wrappers import ReacherWrapper

PlannerStatus = {
    'Exact solution': 1,
    'Approximate solution': 2,
}


class ReacherIRLObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn: Callable, goal: np.ndarray):
        super().__init__(si)
        self.cost_fn = cost_fn
        self.goal = goal

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        """Query the neural network cost function for a cost between two states"""
        s1_np = convert_ompl_state_to_numpy(s1)
        s2_np = convert_ompl_state_to_numpy(s2)
        s1_np = np.concatenate([s1_np, self.goal])
        s2_np = np.concatenate([s2_np, self.goal])

        c = self.cost_fn(s1_np, s2_np)
        return ob.Cost(c)

class MinimumTransitionObjective(ob.PathLengthOptimizationObjective):
    """Minimum number of transitions"""

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super().__init__(si)

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        return ob.Cost(1.0)

class ReacherShortestDistanceObjective(ob.PathLengthOptimizationObjective):
    """
    Cost for a state is its distance to target
    OMPL does not allow control cost, thus ignoring the control effort here
    """

    def __init__(self, si: Union[oc.SpaceInformation, ob.SpaceInformation]):
        super().__init__(si)

    def stateCost(self, s: ob.State) -> ob.Cost:
        target = s[3][:2]
        finger = s[4][:2]
        c = np.linalg.norm(target - finger)
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

def convert_ompl_state_to_numpy(state: ob.State) -> np.ndarray:
    return np.array([state[0].value, state[1].value, state[2][0], state[2][1], state[3][0], state[3][1]])

def path_to_numpy(path: Union[og.PathGeometric, oc.PathControl], dim: int) -> np.ndarray:
    """Convert OMPL path to numpy array"""
    states = np.fromstring(
        path.printAsMatrix(), 
        dtype=np.float32, 
        sep="\n"
    ).reshape(-1, dim)
    return states

def states_to_numpy(states: List[ob.State]) -> np.ndarray:
    """Convert OMPL controls to numpy array"""
    return np.stack([convert_ompl_state_to_numpy(state) for state in states]).astype(np.float32)

def controls_to_numpy(controls: List[oc.Control], dim: int) -> List[np.ndarray]:
    """Convert OMPL controls to numpy array"""
    controls_np = np.zeros((len(controls), dim), dtype=np.float32)
    for i in range(len(controls)):
        for j in range(dim):
            controls_np[i][j] = controls[i][j]
    return controls_np


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

##################################################################


def plan_from_states(
    states: th.Tensor,
    cost_fn: Callable[[np.ndarray], np.ndarray]
) -> List[th.Tensor]:
    """Construct planner instance for each start location"""
    states = [ptu.to_numpy(state) for state in states]
    # args = [[state, cost_fn] for state in states]
    # with mp.Pool(os.cpu_count()-1) as pool:
    #     results = pool.starmap(plan_from_state, args)
    # status, paths, controls = list(zip(*results))
    # paths = [ptu.from_numpy(path) for path in paths]

    paths = []
    for state in states:
        status, path, control = plan_from_state(state, cost_fn)
        paths.append(path)
    paths = [ptu.from_numpy(path) for path in paths]
    return paths

def plan_from_state(
    state: np.ndarray,
    cost_fn: Callable[[np.ndarray], np.ndarray]
) -> Tuple[str, np.ndarray, np.ndarray]:
    # Construct env and planner
    start, goal = state[:6].astype(np.float64), state[6:].astype(np.float64)

    # env = ReacherWrapper(gym.make("Reacher-v2"))
    # planner = cp.ReacherSSTPlanner(env)
    planner = gp.ReacherRRTstarPlanner()
    planner.update_ss_cost(cost_fn, goal)

    status, path, control = planner.plan(start, goal)
    assert status in PlannerStatus.keys(), f"Planner failed with status {status}"
    # assert len(path) == len(control) + 1, (
    #     f"Path length {len(path)} does not match control length {len(control)}"
    # )

    # Need to pad target position back to each state
    path = np.concatenate((
        path,
        np.repeat(goal.reshape(1, -1), len(path), axis=0)
    ), axis=1)
    assert path.shape[1] == state.shape[0]
    return status, path, control

def add_states_to_paths(
    states: th.Tensor, 
    paths: th.Tensor
) -> List[th.Tensor]:
    """Add initial states to path"""
    assert len(states) == len(paths), (
        f"Lengths of state {len(states)} and paths {len(paths)} are not equal"
    )
    padded_paths = [
        th.cat((state.reshape(1, -1), path), dim=0) 
        for state, path in zip(states, paths)
    ]
    return padded_paths

