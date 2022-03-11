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
import irl.utils.pytorch_utils as ptu
from irl.utils.wrappers import ReacherWrapper

PlannerStatus = {
    'Exact solution': 1,
    'Approximate solution': 2,
}

def compute_xy_from_angles(th1: float, th2: float) -> Tuple[float, float]:
    """
    Compute fingertip xy given the angles of the two joints
    arm length is 0.1 each, finger ball size is 0.01
    second angle is w.r.t. first angle
    """
    # assert (-np.pi <= th1 <= np.pi and -np.pi <= th2 <= np.pi), (
    #     f"Angles {th1, th2} not in range -pi to pi"
    # )
    th2 = th2 + th1
    xy1 = np.array([np.cos(th1), np.sin(th1)]) * 0.1
    xy2 = np.array([np.cos(th2), np.sin(th2)]) * 0.11
    xy = xy1 + xy2
    return xy[0], xy[1]

def compute_angles_from_xy(x: float, y: float) -> Tuple[float, float]:
    """
    Compute the two joint angles such that the reacher fingertip touches the target xy
    We have two unique solutions which are mirror images w.r.t. the line through xy
    Numerical precision 
    """ 

    # Normalize x, y to within 0.21 from origin
    if x**2 + y**2 > 0.21**2:
        l = np.sqrt(x**2 + y**2)
        x = x / l * 0.21
        y = y / l * 0.21

    # Degenerate slope if y = 0
    if np.abs(y) < 1e-6:
        x1 = .1 / .21 * x
        y1 = np.sqrt(0.21**2 - x1**2)
        th1 = np.arctan2(y1, x1)
    else:
        # Compute the perpendicular line of (0, 0) and (x, y) intersecting at 0.1 / 0.21 * (x, y)
        k = - x / y
        b = .1 / .21 * (y + x**2 / y)   
        # assert np.abs(k * .1/.21 * x + b - .1/.21*y) < 1e-6

        # Compute the intersection of the perpendicular bisector and the circle of radius 0.1 at origin
        # Solve to quadratic equation
        coeffs = [k**2 + 1, 2*k*b, b**2 - 0.1**2]
        # x1, x11 = np.roots(coeffs)  
        x1 = (-coeffs[1] + np.sqrt(coeffs[1]**2 - 4*coeffs[0]*coeffs[2])) / (2 * coeffs[0])
        # assert np.linalg.norm(x1-x1_1) < 1e-6 or np.linalg.norm(x1-x1_2) < 1e-6

        # Compute endpoint of first arm (x1, y1) and angle th1
        y1 = k * x1 + b
        th1 = np.arctan2(y1, x1)
        if np.isnan(th1):
            import ipdb; ipdb.set_trace()
    # assert np.abs(x1**2 + y1**2 - 0.1**2) <= 1e-6, f"First arm radius is {np.sqrt(x1**2 + y1**2)}, not 0.1"
    # print(f"computed body1 {x1:.6f}, {y1:.6f}")

    # Compute angle of second arm
    x2, y2 = x - x1, y - y1
    th2 = angle_normalize(np.arctan2(y2, x2) - th1)

    xy_recovered = np.array(compute_xy_from_angles(th1, th2))
    xy = np.array([x, y])
    # assert np.linalg.norm(xy - xy_recovered) < 1e-3, (
    #     f"Given xy {xy}, recovered_xy {xy_recovered}, distance {np.linalg.norm(xy-xy_recovered):.5f}")
    return th1, th2


class ReacherIRLObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn: Callable, target: np.ndarray):
        super().__init__(si)
        self.cost_fn = cost_fn
        self.target = target

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        """Query the neural network cost function for a cost between two states"""
        s1_np = convert_ompl_state_to_numpy(s1)
        s2_np = convert_ompl_state_to_numpy(s2)
        s1_np = np.concatenate([s1_np, self.target])
        s2_np = np.concatenate([s2_np, self.target])

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
    Cost for a state is its distance from fingertip to target, plus angular velocities
    OMPL does not allow control cost, thus ignoring the control effort here
    """
    def __init__(
        self, 
        si: Union[oc.SpaceInformation, ob.SpaceInformation],
        target: np.ndarray
    ):
        super().__init__(si)
        self.target = target

    def stateCost(self, s: ob.State) -> ob.Cost:
        fingertip = compute_xy_from_angles(s[0].value, s[1].value)
        c = np.linalg.norm(self.target - fingertip).item()
        c += np.sqrt(s[2][0] ** 2 + s[2][1] ** 2)
        return ob.Cost(c)

def angle_normalize(x):
    """Normalize angle between -pi and pi"""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

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
    return np.array([state[0].value, state[1].value, state[2][0], state[2][1]])

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
# Planning
##################################################################
# def next_states_from_env(
#     env: gym.Env, 
#     states: th.Tensor, 
#     actions_l: List[th.Tensor]
# ) -> List[th.Tensor]:
#     """Query the environment for next states"""
#     states = ptu.to_numpy(states)
#     actions_l = [ptu.to_numpy(actions) for actions in actions_l]
#     next_states_l = []
#     for actions in actions_l:
#         assert len(states) == len(actions), "Sampled actions not equal to states"
#         next_states = []
#         for state, action in zip(states, actions):
#             next_states.append(env.one_step_transition(state, action))
#         next_states = ptu.from_numpy(np.stack(next_states))
#         assert next_states.shape == states.shape, "Sampled next states not equal to states"
#         next_states_l.append(next_states)
#     return next_states_l

def next_states_from_env(
    env: gym.Env, 
    states: th.Tensor, 
    actions: th.Tensor
) -> th.Tensor:
    """Query the environment for next states"""
    states = ptu.to_numpy(states)
    actions = ptu.to_numpy(actions)
    next_states = []
    for state, action in zip(states, actions):
        next_states.append(env.one_step_transition(state, action))
    return ptu.from_numpy(np.stack(next_states))

def plan_from_states(
    planner: gp.ReacherGeometricPlanner,
    states: th.Tensor,
    cost_fn: Callable[[np.ndarray, np.ndarray], float],
    solveTime: Optional[float] = 1.0,
) -> List[th.Tensor]:
    """Construct planner instance for each start location"""
    states = [ptu.to_numpy(state) for state in states]
    paths = []
    for state in states:
        status, path, control = plan_from_state(planner, state, cost_fn, solveTime)
        paths.append(path)
    paths = [ptu.from_numpy(path) for path in paths]
    return paths

def plan_from_state(
    planner: gp.ReacherGeometricPlanner,
    state: np.ndarray,
    cost_fn: Callable[[np.ndarray, np.ndarray], float],
    solveTime: Optional[float] = 1.0,
) -> Tuple[str, np.ndarray, np.ndarray]:
    # Construct env and planner
    start = state[:4].astype(np.float64) 
    target = state[-2:].astype(np.float64)

    # Each planning problem has a different goal, need to update cost function
    planner.update_ss_cost(cost_fn, target)
    status, path, control = planner.plan(start, target, solveTime=solveTime)
    assert status in PlannerStatus.keys(), f"Planner failed with status {status}"
    # Check planned path makes fingertip reach target xy
    # finger_pos = compute_xy_from_angles(path[-1][0], path[-1][1])
    # dist = np.linalg.norm(target - finger_pos)
    # assert dist < 1e-1, f"Final state fingertip distance to target {dist:.3f}"

    # Need to pad target position back to each state
    path = np.concatenate((
        path,
        np.repeat(target.reshape(1, -1), len(path), axis=0)
    ), axis=1)
    assert path.shape[1] == 6           # 2 pos + 2 vel + 2 target xy
    return status, path, control

def add_states_to_paths(
    states: th.Tensor, 
    paths: th.Tensor
) -> List[th.Tensor]:
    """Add initial states to path"""
    assert len(states) == len(paths), (
        f"Lengths of state {len(states)} and paths {len(paths)} are not equal"
    )
    # states = th.cat([states[:,:4], states[:,-2:]], dim=1)
    padded_paths = [
        th.cat((state.reshape(1, -1), path), dim=0) 
        for state, path in zip(states, paths)
    ]
    return padded_paths

def fixed_horizon_paths(
    paths: List[th.Tensor],
    T: int
) -> th.Tensor:
    fixed_paths = []
    for path in paths:
        if len(path) >= T:
            fixed_paths.append(path[:T])
        else:
            padded = th.tile(path[-1], dims=(T-len(path), 1))
            fixed_paths.append(th.cat((path, padded), dim=0))
    fixed_paths = th.stack(fixed_paths, dim=0)
    return fixed_paths