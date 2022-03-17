from typing import Optional, Tuple

import gym
import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from gcl.planners.base_planner import ReacherBasePlanner
from gcl.utils import planner_utils

np.set_printoptions(precision=3)

class ReacherStatePropagator(oc.StatePropagator):
    """State propagator function for Reacher-v2 environment"""
    def __init__(
        self,
        si: oc.SpaceInformation,
        state_validity_checker: ob.StateValidityChecker,
        state_space: ob.StateSpace,
        env: gym.Env
    ):
        super().__init__(si)

        # assert si.getMinControlDuration() == si.getMaxControlDuration() == 1, (
        #     "SpaceInformation control duration is not set to 2"
        # ) 
        self.state_space = state_space
        self.env = env
        self.state_validity_checker = state_validity_checker

        # A placeholder for qpos and qvel in propagte function that don't waste time on numpy creation
        self.qpos_temp = np.zeros(4)
        self.qvel_temp = np.zeros(4)
        self.ctrl_temp = np.zeros(2)

    def propagate(
        self,
        state: ob.State,
        control: oc.Control,
        duration: float,
        result: ob.State
    ) -> None:
        """
        Propagate the state with control for a duration to the result state.
        We pass the state and control to the mujoco simulator step function
        to query the next state. Duration variable is not used since we are interested
        in computing the next state after one step transition.
        The result state should correspond to the next state in mujoco simulator as
        we assume si.MinMaxControlDuration=1 and env.frame_skip=2
        """
        assert self.state_validity_checker.isValid(state), (
            f"State {state} is not valid before propagate function"
        )
        assert (-1 <= control[0] <= 1 and -1 <= control[1] <= 1), (
            f"Control {control} is not valid before propagate function"
        )

        # ==== Get qpos and qvel from ompl state ====
        # We only populate the first two dimensions of qpos and qvel, 
        # which corresponds to the two joints of the reacher
        # The last two dimensions of qpos and qvel (for target) are zero 
        self.qpos_temp[0] = state[0].value
        self.qpos_temp[1] = state[1].value
        self.qvel_temp[0] = state[2][0]
        self.qvel_temp[1] = state[2][1]
        self.ctrl_temp[0] = control[0]
        self.ctrl_temp[1] = control[1]

        # ==== Propagate qpos and qvel with given control in Mujoco===
        # assume MinMaxControlDuration = 1 and frame_skip = 2
        self.env.set_state(self.qpos_temp, self.qvel_temp)
        self.env.step(self.ctrl_temp)
        # self.env.do_simulation(self.ctrl_temp, self.env.frame_skip)

        # ==== Copy Mujoco State to OMPL State ====
        qpos = self.env.sim.data.qpos.flat[:].copy()
        qvel = self.env.sim.data.qvel.flat[:].copy()
        fingertip = self.env.get_body_com("fingertip")[:2]

        result[0].value = qpos[0]
        result[1].value = qpos[1]
        result[2][0] = qvel[0]
        result[2][1] = qvel[1]
        result[3][0] = fingertip[0]
        result[3][1] = fingertip[1]

        self.state_space.enforceBounds(result)
        # ==== End of propagate ====

        assert self.state_validity_checker.isValid(result), (
            f"Result state {planner_utils.convert_ompl_state_to_numpy(result)} ",
            "is not valid after propagate function"
        )
    
    def canPropagateBackward(self) -> bool:
        """Does not allow backward state propagation in time"""
        return False


class ReacherControlPlanner(ReacherBasePlanner):
    """
    Parent class for SST and RRT control planners
    Instantiate StateSpace, ControlSpace, SimpleSetup, 
    SpaceInformation, OptimizationObjective, etc
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
        self.state_propagator = self.get_StatePropagator(
            self.si, 
            self.state_validity_checker,
            self.space,
            env 
        )
        self.ss.setStatePropagator(self.state_propagator)

        # Define optimization objective
        self.objective = planner_utils.ReacherShortestDistanceObjective(self.si)
        self.ss.setOptimizationObjective(self.objective)

    def get_ControlSpace(self, space: ob.StateSpace) -> oc.ControlSpace:
        """
        Create the control space for Reacher-v2
        Control is torque
        """
        # Set parameters for state space
        control_low = np.array([-1., -1.], dtype=np.float32)
        control_high = np.array([1., 1.], dtype=np.float32)

        #########################################
        cspace = oc.RealVectorControlSpace(space, self.control_dim)
        c_bounds = planner_utils.make_RealVectorBounds(
            dim=self.control_dim,
            low=control_low,
            high=control_high
        )
        cspace.setBounds(c_bounds)

        return cspace

    def get_StatePropagator(
        self,
        si: oc.SpaceInformation,
        state_validity_checker: ob.StateValidityChecker,
        state_space: ob.StateSpace,
        env: gym.Env
    ) -> oc.StatePropagator:
        return ReacherStatePropagator(si, state_validity_checker, state_space, env)

    def plan(
        self, 
        start: np.ndarray, 
        goal: np.ndarray,
        solveTime: Optional[float] = 1.0,
        total_solveTime: Optional[float] = 20.0
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Return a list of states and controls
        """
        # Clear previous planning data and set new start state
        self.ss.clear()
        self.ss.setStartState(self.get_StartState(start))
        self.ss.setGoal(self.get_Goal(self.si, goal))

        status = self.ss.solve(solveTime)
        t = self.ss.getLastPlanComputationTime()

        while planner_utils.PlannerStatus[status.asString()] != 1 and t <= total_solveTime:
            status = self.ss.solve(solveTime)
            t += self.ss.getLastPlanComputationTime() 

        msg = planner_utils.color_status(status)
        objective = self.ss.getProblemDefinition().getOptimizationObjective()

        if bool(status):
            # Retrieve path and controls
            control_path = self.ss.getSolutionPath()
            control_path.interpolate()
            geometric_path = control_path.asGeometric()
            states = control_path.getStates()
            controls = control_path.getControls()
            print(
                f"{msg}: "
                f"Path length is {geometric_path.length():.2f}, "
                f"cost is {geometric_path.cost(objective).value():.2f}, "
                f"solve time is {t:.4f}"
            )
            # Convert to numpy arrays
            # states = planner_utils.path_to_numpy(geometric_path, dim=self.state_dim)
            states = planner_utils.states_to_numpy(states)
            controls = planner_utils.controls_to_numpy(controls, dim=self.control_dim)
            return status.asString(), states, controls
        else:
            print(status.asString())
            raise ValueError("OMPL is not able to solve under current cost function")

class ReacherSSTPlanner(ReacherControlPlanner):
    def __init__(self, env):
        super().__init__(env)

        planner = oc.SST(self.si)
        #TODO: check planner selection/pruning radius
        self.ss.setPlanner(planner)


