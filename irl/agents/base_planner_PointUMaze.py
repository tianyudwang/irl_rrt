from math import pi, sin, cos
from typing import Union, Tuple


import numpy as np

from mujoco_maze.agent_model import AgentModel

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

from icecream import ic


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


class IRLCostObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_fn):
        super(IRLCostObjective, self).__init__(si)
        self.cost_fn = cost_fn

    def motionCost(self, s1, s2):
        x1, y1 = s1[0].getX(), s1[0].getY()
        x2, y2 = s2[0].getX(), s2[0].getY()
        s1 = np.array([x1, y1], dtype=np.float64)
        s2 = np.array([x2, y2], dtype=np.float64)
        c = self.cost_fn(s1, s2)
        return ob.Cost(c)


def getIRLCostObjective(si, cost_fn) -> ob.OptimizationObjective:
    return IRLCostObjective(si, cost_fn)


class MazeGoal(ob.Goal):
    """
    An OMPL Goal that satisfy when euclidean dist between state and goal under a threshold
    """

    def __init__(self, si: oc.SpaceInformation, goal: np.ndarray, threshold: float):
        super().__init__(si)
        assert goal.ndim == 1
        self.si = si
        self.goal = goal[:2]
        self.threshold = threshold

    def isSatisfied(self, state: ob.State) -> bool:
        """
        Check if the state is the goal.
        """
        SE2_state = state[0]
        x, y = SE2_state.getX(), SE2_state.getY()
        return np.linalg.norm(self.goal - np.array([x, y])) < self.threshold


class PointStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si: Union[oc.SpaceInformation, ob.SpaceInformation],
    ):
        super().__init__(si)
        self.si = si
        self.size = 0.5
        self.scaling = 4.0
        self.x_limits = [-2, 10]
        self.y_limits = [-2, 10]

        self.Umaze_x_min = self.x_limits[0] + self.size
        self.Umaze_y_min = self.y_limits[0] + self.size
        self.Umaze_x_max = self.x_limits[1] - self.size
        self.Umaze_y_max = self.y_limits[1] - self.size

    def isValid(self, state: ob.State) -> bool:

        SE2_state = state[0]
        # assert isinstance(SE2_state, ob.SE2StateSpace.SE2StateInternal)

        x_pos = SE2_state.getX()
        y_pos = SE2_state.getY()

        # In big square contains U with point size constrained
        inSquare = all(
            [
                self.Umaze_x_min <= x_pos <= self.Umaze_x_max,
                self.Umaze_y_min <= y_pos <= self.Umaze_y_max,
            ]
        )
        if inSquare:
            # In the middle block cells
            inMidBlock = all(
                [
                    self.x_limits[0] <= x_pos <= 6.5,  # 6 + self.size,
                    1.5 - self.size <= y_pos <= 6.5,  #  2 - self.size, 6 + self.size,
                ]
            )
            if inMidBlock:
                valid = False
            else:
                valid = True
        # Not in big square
        else:
            valid = False

        # Inside empty cell and satisfiedBounds
        return valid and self.si.satisfiesBounds(state)


class PointStatePropagator(oc.StatePropagator):
    def __init__(
        self,
        si: oc.SpaceInformation,
        agent_model: AgentModel,
        velocity_limits: float = 10,
    ):
        super().__init__(si)
        self.si = si
        self.agent_model = agent_model
        self.velocity_limits = velocity_limits

        self.bounds_low = [-2, -2, -np.pi, -12, -12, -12]
        self.bounds_high = [10, 10, np.pi, 12, 12, 12]

        # A placeholder for qpos and qvel in propagte function that don't waste tme on numpy creation
        self.qpos_temp = np.empty(3)
        self.qvel_temp = np.empty(3)

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        """
        propagate function for control planning
        Note duration is a dummy placeholder. 
        """
        # Control [ballx, rot]
        assert self.si.satisfiesBounds(state), "Input state not in bounds"
        # SE2_state: qpos = [x, y, Yaw]
        SE2_state = state[0]
        # V_state: qvel = [vx, vy, w]
        V_state = state[1]

        self.qpos_temp[0] = SE2_state.getX()
        self.qpos_temp[1] = SE2_state.getY()
        self.qpos_temp[2] = SE2_state.getYaw()

        self.qvel_temp[0] = V_state[0]
        self.qvel_temp[1] = V_state[1]
        self.qvel_temp[2] = V_state[2]

        self.qpos_temp[2] += control[1]
        # Check if the orientation is in [-pi, pi],
        # if not normalize it to be in [-pi, pi], since it is SO2
        if not (-pi <= self.qpos_temp[2] <= pi):
            self.qpos_temp[2] = angle_normalize(self.qpos_temp[2])

        # Compute increment in each direction
        ori = self.qpos_temp[2]
        self.qpos_temp[0] += cos(ori) * control[0]
        self.qpos_temp[1] += sin(ori) * control[0]

        # Clip velocity is encode in cbound. range from [-12, 12] to [-10, 10] instead of clipping.

        # copy OMPL State to Mujoco
        self.agent_model.set_state(self.qpos_temp, self.qvel_temp)

        # assume MinMaxControlDuration = 1 and frame_skip = 1
        self.agent_model.sim.step()
        next_obs = self.agent_model._get_obs()

        # Yaw angle migh be out of range [-pi, pi] after several propagate steps.
        # Should enforced yaw angle since it is SO2 and should be always in bounds
        if not (-pi <= next_obs[2] <= pi):
            next_obs[2] = angle_normalize(next_obs[2])

        # Copy Mujoco State to OMPL
        # next SE2_state: next_qpos = [x, y, Yaw]
        result[0].setX(next_obs[0])
        result[0].setY(next_obs[1])
        result[0].setYaw(next_obs[2])

        # next V_state: next_qvel = [vx, vy, w]
        result[1][0] = next_obs[3]
        result[1][1] = next_obs[4]
        result[1][2] = next_obs[5]

    def canPropagateBackward(self) -> bool:
        return False

    def canSteer(self) -> bool:
        return False


class BasePlannerPointUMaze:
    def __init__(self, env, use_control=False, log_level=0):
        # TODO: the env might have several wrapper
        self.env = env
        self.agent_model = self.env.unwrapped.wrapped_env

        self.init_sim_state = self.agent_model.sim.get_state()
        self.start = np.concatenate(
            [self.init_sim_state.qpos, self.init_sim_state.qvel]
        )

        # space information
        self.state_dim = 6
        self.control_dim = 2

        # state bound
        self.state_high = np.array([10, 10, pi, 10, 10, 10])
        self.state_low = -self.state_high
        # control bound
        self.control_high = np.array([1, 0.25])
        self.control_low = -self.control_high
        
        # goal position and goal radius
        self.goal = np.array([0, 8], dtype=np.float64)
        self.threshold = 0.6

        self.qpos_low = self.state_low[:2]
        self.qpos_high = self.state_high[:2]
        self.qvel_low = self.state_low[3:]
        self.qvel_high = self.state_high[3:]

        self.init_simple_setup(use_control, log_level)

    @property
    def PropagStepSize(self) -> float:
        return self.agent_model.sim.model.opt.timestep

    def makeStateSpace(self, lock: bool = True) -> ob.StateSpace:
        """
        Create a state space.
        """
        # State Space (A compound space include SO3 and accosicated velocity).
        # SE2 = R^2 + SO2. Should not set the bound for SO2 since it is enfored automatically.
        SE2_space = ob.SE2StateSpace()
        SE2_bounds = make_RealVectorBounds(
            bounds_dim=2,
            low=self.qpos_low,
            high=self.qpos_high,
        )
        SE2_space.setBounds(SE2_bounds)

        # velocity space.
        velocity_space = ob.RealVectorStateSpace(3)
        v_bounds = make_RealVectorBounds(
            bounds_dim=3,
            low=self.qvel_low,
            high=self.qvel_high,
        )
        velocity_space.setBounds(v_bounds)

        # Add subspace to the compound space.
        space = ob.CompoundStateSpace()
        space.addSubspace(SE2_space, 1.0)
        space.addSubspace(velocity_space, 1.0)

        # Lock this state space. This means no further spaces can be added as components.
        if space.isCompound() and lock:
            space.lock()
        return space

    def makeControlSpace(self, state_space: ob.StateSpace) -> oc.ControlSpace:
        """
        Create a control space and set the bounds for the control space
        """
        cspace = oc.RealVectorControlSpace(state_space, 2)
        c_bounds = make_RealVectorBounds(
            bounds_dim=2,
            low=self.control_low,
            high=self.control_high,
        )
        cspace.setBounds(c_bounds)
        return cspace

    def makeStartState(self) -> ob.State:
        """
        Create a start state.
        """
        if isinstance(self.start, np.ndarray):
            assert self.start.ndim == 1
        start = ob.State(self.space)
        for i in range(len(self.start)):
            start[i] = self.start[i]
        return start

    def init_simple_setup(self, use_control: bool = False, log_level: int = 0):
        """
        Initialize an ompl::control::SimpleSetup instance
            or ompl::geometric::SimpleSetup instance. if use_control is False.
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

        # Define State Space
        self.space = self.makeStateSpace()

        if use_control:
            self.cspace = self.makeControlSpace(self.space)
            # Define a simple setup class
            self.ss = oc.SimpleSetup(self.cspace)
            # Retrieve current instance of Space Information
            self.si = self.ss.getSpaceInformation()
            # Set the start state and goal state
            start = self.makeStartState()
            goal = MazeGoal(self.si, self.goal, self.threshold)
            self.ss.setStartState(start)
            self.ss.setGoal(goal)

            # Set State Validation Checker
            stateValidityChecker = PointStateValidityChecker(self.si)
            self.ss.setStateValidityChecker(stateValidityChecker)

            # Set State Propagator
            propagator = PointStatePropagator(self.si, self.agent_model)
            self.ss.setStatePropagator(propagator)

            # Set propagator step size
            # ? PropagationStepSize refer to duration in propagation fucntion which is not using.
            self.si.setPropagationStepSize(self.PropagStepSize)  # 0.02 in Mujoco
            self.si.setMinMaxControlDuration(minSteps=1, maxSteps=1)

        else:
            self.cspace = None
            # TODO: og.SimpleSetup
            pass

    def update_ss_cost(self, cost_fn):
        # Set up cost function
        costObjective = getIRLCostObjective(self.si, cost_fn)
        self.ss.setOptimizationObjective(costObjective)

    def plan(self, start_state: np.ndarray, solveTime: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform motion planning for a specified amount of time.
        """
        # Clear previous planning computation data, does not affect settings and start/goal
        self.ss.clear()

        # Reset the start state
        start = ob.State(self.space)
        start[0], start[1] = start_state[0].item(), start_state[1].item()
        self.ss.setStartState(start)

        solved = self.ss.solve(solveTime)
        if solved:
            control_path = self.ss.getSolutionPath()
            states = np.asarray(
                [
                    [
                        state[0].getX(),
                        state[0].getY(),
                        state[0].getYaw(),
                        state[1][0],
                        state[1][1],
                        state[1][2],
                    ]
                    for state in control_path.getStates()
                ],
                dtype=np.float32,
            )
            controls = np.asarray(
                [[u[0], u[1]] for u in control_path.getControls()], dtype=np.float32
            )
            return states, controls
        else:
            raise ValueError("OMPL is not able to solve under current cost function")


class SSTPlanner(BasePlannerPointUMaze):
    def __init__(self, env):
        super(SSTPlanner, self).__init__(env, use_control=True, log_level=1)
        self.init_planner()

    def init_planner(self):
        # Set planner
        planner = oc.SST(self.si)
        self.ss.setPlanner(planner)


if __name__ == "__main__":
    import gym
    import mujoco_maze

    env = gym.make("PointUMaze-v0")
    env.reset()
    planner = SSTPlanner(env)

    planner.ss.setPlanner(oc.RRT(planner.si))
    _, controls = planner.plan(planner.start, solveTime=5)

    # Ensure we have the same start position
    env.unwrapped.wrapped_env.sim.set_state(planner.init_sim_state)
    for u in controls:
        qpos = env.unwrapped.wrapped_env.sim.data.qpos
        qvel = env.unwrapped.wrapped_env.sim.data.qvel
        # print(f"qpos: {qpos}, qvel: {qvel}")

        obs, rew, done, info = env.step(u)

        if done:
            print("Reach Goal. Success!!")
            break
    env.close()
    # test pass
