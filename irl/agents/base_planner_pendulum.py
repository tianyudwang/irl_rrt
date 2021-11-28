from math import pi, sin, cos
from functools import partial
from typing import Union

import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

## Defines an optimization objective by computing the cost of motion between
# two endpoints.
class IRLCostObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super(IRLCostObjective, self).__init__(si)
        self.cost_fn = cost_fn

    def motionCost(self, s1, s2):
        s1 = np.array([s1[0].value, s1[1][0]], dtype=np.float32)
        s2 = np.array([s2[0].value, s2[1][0]], dtype=np.float32)
        c = self.cost_fn(s1, s2)
        return ob.Cost(c)


def getIRLCostObjective(si, cost_fn):
    return IRLCostObjective(si, cost_fn)


class BasePlanner:
    def __init__(self):
        # Space information
        self.state_dim = 2
        self.state_low = np.array([-pi, -8])
        self.state_high = np.array([pi, 8])
        self.control_low = -2.0
        self.control_high = 2.0
        self.goal = np.array([0.0, 0.0], dtype=np.float64)

        # Pendulum parameters
        g = 10.0
        m = 1.0
        l = 1.0
        self.dt = 0.05
        self.max_angular_velocity = 8.0
        self.max_torque = 2.0

        self.a = 3.0 * g / (2.0 * l)
        self.b = 3.0 / (m * l ** 2)

        self.init_simple_setup()

    def construct_spaces(self):
        # Construct [theta, theta_dot] state space
        # SO2 state space enforces angle to be in [-pi, pi]
        th_space = ob.SO2StateSpace()
        th_dot_space = ob.RealVectorStateSpace(1)
        th_dot_bounds = ob.RealVectorBounds(1)
        th_dot_bounds.setLow(self.state_low[1])
        th_dot_bounds.setHigh(self.state_high[1])
        th_dot_space.setBounds(th_dot_bounds)

        # Create compound space which allows the composition of state spaces.
        space = ob.CompoundStateSpace()
        space.addSubspace(th_space, 1.0)
        space.addSubspace(th_dot_space, 1.0)
        # Lock this state space. This means no further spaces can be added as components.
        space.lock()

        # Create a control space
        cspace = oc.RealVectorControlSpace(space, 1)
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(self.control_low)
        cbounds.setHigh(self.control_high)
        cspace.setBounds(cbounds)

        return space, cspace

    def isStateValid(self, si, state: ob.State):
        """perform collision checking or check if other constraints are satisfied"""
        return si.satisfiesBounds(state)

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ):
        """
        Define the discrete time dynamics.
        Computes the next state given current state, control, control duration.
        """
        th, th_dot, u = state[0].value, state[1][0], control[0]

        # Assert states are proper
        assert -pi <= th <= pi, f"State theta is out of bounds: {th}"
        assert -8.0 <= th_dot <= 8.0, f"State theta_dot is out of bounds: {th_dot}"
        assert -2.0 <= u <= 2, f"Control input u is out of bounds: {u}"

        # newthdot = th_dot + (3.0 * self.g / (2.0 * self.l) * sin(th)
        #                      + 3.0 / (self.m * self.l ** 2) * u) * duration
        newthdot = th_dot + (self.a * sin(th) + self.b * u) * duration
        newthdot = np.clip(
            newthdot, -self.max_angular_velocity, self.max_angular_velocity
        )
        newth = th + newthdot * duration

        result[0].value = newth
        result[1][0] = newthdot

        # Enforce the angle in SO2
        self.space.enforceBounds(result)

    def init_simple_setup(self, log_level=0):
        """
        Initialize an ompl::control::SimpleSetup instance
        """
        assert isinstance(log_level, int)
        assert 0 <= log_level <= 2

        # Set log to warn/info/debug
        if log_level == 0:
            ou.setLogLevel(ou.LOG_WARN)
        elif log_level == 1:
            ou.setLogLevel(ou.LOG_INFO)
        else:
            ou.setLogLevel(ou.LOG_DEBUG)

        # Define state and control spaces
        self.space, self.cspace = self.construct_spaces()

        # Define a simple setup class
        ss = oc.SimpleSetup(self.cspace)
        self.si = ss.getSpaceInformation()
        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(partial(self.isStateValid, self.si))
        )
        ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        # Set the agent goal state
        goal = ob.State(self.space)
        goal[0], goal[1] = self.goal[0], self.goal[1]
        ss.setGoalState(goal)
        self.ss = ss

        # Set propagation step size -> duration of each step
        self.si.setMinMaxControlDuration(1, 1)
        self.si.setPropagationStepSize(self.dt)

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)

    def plan(self, start_state, solveTime=0.5):
        # Clear previous planning data, does not affect settings and start/goal
        self.ss.clear()

        # Reset the start state
        start = ob.State(self.space)
        start[0], start[1] = start_state[0].item(), start_state[1].item()
        self.ss.setStartState(start)

        # Solve and get optimal path
        #        while not self.ss.getProblemDefinition().hasExactSolution():
        #            solved = self.ss.solve(solveTime)
        solved = self.ss.solve(5.0)
        if solved:
            control_path = self.ss.getSolutionPath()
            states = np.asarray(
                [[state[0].value, state[1][0]] for state in control_path.getStates()],
                dtype=np.float32,
            )
            controls = np.asarray(
                [u[0] for u in control_path.getControls()], dtype=np.float32
            )
            return states, controls
        else:
            raise ValueError("OMPL is not able to solve under current cost function")
            return None, None


class SSTPlanner(BasePlanner):
    def __init__(self):
        super(SSTPlanner, self).__init__()
        self.init_planner()

    def init_planner(self):
        # Set planner
        planner = oc.SST(self.si)
        self.ss.setPlanner(planner)
