from typing import Optional, Tuple

import numpy as np
import gym   

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from irl.planners.base_planner import Maze2DBasePlanner
import irl.planners.planner_utils as planner_utils


class Maze2DStatePropagator(oc.StatePropagator):
    """State propagator function for maze2d-umaze-v1 environment"""
    def __init__(
        self,
        si: oc.SpaceInformation,
        state_validity_checker: ob.StateValidityChecker,
        env: gym.Env
    ):
        super().__init__(si)

        assert si.getMinControlDuration() == si.getMaxControlDuration()== 1, (
            "SpaceInformation control duration is not set to 1"
        ) 
        assert env.frame_skip == 1, "Mujoco env frame_skip is not set to 1"

        self.env = env
        self.state_validity_checker = state_validity_checker

        # A placeholder for qpos and qvel in propagte function that don't waste time on numpy creation
        self.qpos_temp = np.empty(2)
        self.qvel_temp = np.empty(2)
        self.ctrl_temp = np.empty(2)

    def propagate(
        self,
        state: ob.State,
        control: oc.Control,
        duration: float,
        result: ob.State
    ):
        """
        Propagate the state with control for a duration to the result state.
        We pass the state and control to the mujoco simulator step function
        to query the next state. Duration variable is not used since we are interested
        in computing the next state after one step transition.
        The result state should correspond to the next state in mujoco simulator as
        we assume si.MinMaxControlDuration=1 and env.frame_skip=1
        """
        
        assert self.state_validity_checker.isValid(state), (
            f"State {state} is not valid before propagate function"
        )
        assert -1 <= control[0] <= 1 and -1 <= control[1] <= 1, (
            f"Control {control} is not valid before propagate function"
        )

        # ==== Get qpos and qvel from ompl state ====
        for i in range(2):
            self.qpos_temp[i] = state[0][i]
        for j in range(2):
            self.qvel_temp[j] = state[1][j]
        for k in range(2):
            self.ctrl_temp[k] = control[k]

        # ==== Propagate qpos and qvel with given control in Mujoco===
        # assume MinMaxControlDuration = 1 and frame_skip = 1
        self.env.set_state(self.qpos_temp, self.qvel_temp)
        self.env.do_simulation(self.ctrl_temp, self.env.frame_skip)
        # obtain new simulation result
        next_obs = self.env._get_obs()

        # ==== Copy Mujoco State to OMPL State ====
        for p in range(2):
            result[0][p] = next_obs[p]
        for q in range(2):
            result[1][q] = next_obs[2+q]
        # ==== End of propagate ====

        # assert self.state_validity_checker.isValid(result), (
        #     f"Result {[result[0][0], result[0][1], result[1][0], result[1][1]]} "
        #     "is not valid after propagate function"
        # )

    def canPropagateBackward(self) -> bool:
        """Does not allow backward state propagation in time"""
        return False


class Maze2DControlPlanner(Maze2DBasePlanner):
    """
    Parent class for SST and RRT control planners
    Instantiate StateSpace, ControlSpace, SimpleSetup, 
    SpaceInformation, OptimizationObjective, etc
    Requires an unwrapped mujoco env for querying state propagation
    """
    def __init__(self, env):
        super().__init__()

        # First set up StateSpace and ControlSpace and SimpleSetup
        self.space = self.get_StateSpace()
        self.cspace = self.get_ControlSpace(self.space)
        self.ss = oc.SimpleSetup(self.cspace)

        # Add StateValidityChecker to SimpleSetup
        self.si = self.ss.getSpaceInformation()
        self.state_validity_checker = self.get_StateValidityChecker(self.si)
        self.ss.setStateValidityChecker(self.state_validity_checker)

        # Add StatePropagator to SimpleSetup
        self.si.setMinMaxControlDuration(1, 1)
        state_propagator = self.get_StatePropagator(
            self.si, 
            self.state_validity_checker, 
            env
        )
        self.ss.setStatePropagator(state_propagator)

        # Add Goal to SimpleSetup
        goal = self.get_Goal(self.si)
        self.ss.setGoal(goal)

        # Define optimization objective
        objective = planner_utils.ShortestDistanceObjective(self.si)
        self.ss.setOptimizationObjective(objective)

    def get_ControlSpace(self, space: ob.StateSpace) -> oc.ControlSpace:
        """
        Create the control space for maze2d-umaze-v1
        Control includes the change in x and y
        """
        # Set parameters for state space
        control_dim = 2
        control_low = np.array([-1., -1.])
        control_high = np.array([1., 1.])

        #########################################
        cspace = oc.RealVectorControlSpace(space, control_dim)
        c_bounds = planner_utils.make_RealVectorBounds(
            dim=control_dim,
            low=control_low,
            high=control_high
        )
        cspace.setBounds(c_bounds)

        #########################################
        print(f"\nCreated control space {cspace.getName()} with {cspace.getDimension()} dimensions")

        return cspace

    def get_StatePropagator(self, si, state_validity_checker, env):
        return Maze2DStatePropagator(si, state_validity_checker, env)

    def plan(
        self, 
        start: np.ndarray, 
        solveTime: Optional[float] = 5.0
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls (None for geometric planners)
        """
        # Clear previous planning data and set new start state
        self.ss.clear()
        start = self.get_StartState(start)
        self.ss.setStartState(start)

        status = self.ss.solve(solveTime)

        if bool(status):
            # Retrieve path and controls
            control_path = self.ss.getSolutionPath()
            geometric_path = control_path.asGeometric()
            controls = control_path.getControls()

            # Convert to numpy arrays
            states = planner_utils.path_to_numpy(geometric_path)
            controls = planner_utils.controls_to_numpy(controls)
            return planner_utils.PlannerStatus[status.asString()], states, controls
        else:
            raise ValueError("OMPL is not able to solve under current cost function")


class Maze2DSSTPlanner(Maze2DControlPlanner):
    def __init__(self, env):
        super().__init__(env)

        planner = oc.SST(self.si)
        #TODO: check planner selection/pruning radius
        self.ss.setPlanner(planner)

class Maze2DRRTPlanner(Maze2DControlPlanner):
    def __init__(self, env):
        super().__init__(env)

        planner = oc.RRT(self.si)
        #TODO: check planner selection/pruning radius
        self.ss.setPlanner(planner)