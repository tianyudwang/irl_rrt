from math import pi, cos, sin
from copy import deepcopy

import gym
import numpy as np
from icecream import ic


def angle_normalize(x: float) -> float:
    return ((x + pi) % (2 * pi)) - pi


class PointUMazeOneStepTransitionWrapper(gym.Wrapper):
    """
    Wrapper for the environment that allows one step of the environment
    to be taken and the observation returned.
    """

    def __init__(self, env):
        super(PointUMazeOneStepTransitionWrapper, self).__init__(env)

        self.dummy_env = env.unwrapped.wrapped_env
        self._collision = self.unwrapped._collision
        self._restitution_coef = self.unwrapped._restitution_coef

        # Point radius
        self.size = 0.4
        self.Umaze_x_min = self.Umaze_y_min = -2 + self.size
        self.Umaze_x_max = self.Umaze_y_max = 10 - self.size


        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.VELOCITY_LIMITS = 10.0

        self.cbound_high = np.array([1, 0.25])
        self.cbound_low = -self.cbound_high

        self.agent_model = env.unwrapped.wrapped_env
        self.nq = self.agent_model.sim.model.nq
        self.nv = self.agent_model.sim.model.nv

        assert (
            self.nq + self.nv == self.observation_space.shape[0]
        ), f"env's obs space likely has time append as last dim. {self.nq + self.nv}: {self.observation_space.shape[0]}"

    @property
    def frame_skip(self):
        return self.agent_model.frame_skip
        
    def one_step_transition(self, state: np.ndarray, action: np.ndarray):
        """
        Query the env for s', for any given s and a.
        Using mujoco internal:
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py#L117-L124
        """

        
        qpos_temp = state[: self.nq].copy()
        qvel_temp = state[self.nq : self.nq + self.nv].copy()

        assert qpos_temp.shape == (self.nq,)
        assert qvel_temp.shape == (self.nv,)
        assert qpos_temp.shape[0] + qvel_temp.shape[0] == state.shape[0]

        qpos_temp[2] += action[1]
        # Normalize orientation to be in [-pi, pi], since it is SO2
        if not (-pi <= qpos_temp[2] <= pi):
            qpos_temp[2] = angle_normalize(qpos_temp[2])

        # Compute increment in each direction
        ori = qpos_temp[2]
        qpos_temp[0] += cos(ori) * action[0]
        qpos_temp[1] += sin(ori) * action[0]
        
        qvel_temp = np.clip(qvel_temp, -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        self.agent_model.set_state(qpos_temp, qvel_temp)
        for _ in range(0, self.frame_skip):
            self.agent_model.sim.step()
        next_obs = self.agent_model._get_obs()
        
        
        # cheeck if the next state is in bound
        # * here next_obs might be out of bounds. Enforce it
        if  next_obs[2]< -pi or next_obs[2] >pi:
            print("enforce yaw") 
            next_obs[2] = angle_normalize(next_obs[2])        
        
        next_obs[3:] = np.clip(next_obs[3:], -self.VELOCITY_LIMITS, self.VELOCITY_LIMITS)
        assert -pi <= next_obs[2] <= pi, "Yaw out of bounds after mj sim step"
        assert -10 <= next_obs[3] <= 10, "x-velocity out of bounds after mj sim step"
        assert -10 <= next_obs[4] <= 10, "y-velocity out of bounds after mj sim step"
        assert -10 <= next_obs[5] <= 10, "yaw-velocity out of bounds after mj sim step"
        
        old_pos = state[:2].copy()
        new_pos = next_obs[:2].copy()
        
        # Checks if the new_position is in the wall
        collision = self._collision.detect(old_pos, new_pos)
        if collision is not None:
            ic(collision)
            # return state
            adjust_pos = collision.point + self._restitution_coef * collision.rest()
            if self._collision.detect(old_pos, adjust_pos) is not None:
                # If pos is also not in the wall, we give up computing the position
                next_obs = state.copy()
                print("also not in wall, give up computing the position")
            else:
                # self.wrapped_env.set_xy(pos)
                next_obs[:2] = adjust_pos
        return next_obs

    def isValid(self, state: np.ndarray) -> bool:

        x_pos = state[0]
        y_pos = state[1]

        # In big square contains U with point size constrained
        inSquare = all(
            [
                self.Umaze_x_min <= x_pos <= self.Umaze_x_max,
                self.Umaze_y_min <= y_pos <= self.Umaze_y_max,
            ]
        )
        if inSquare:
            # In the middle block cells
            inMidBlock = (-2 <= x_pos <= 6.4) and (1.4 <= y_pos <= 6.4)
            if inMidBlock:
                valid = False
                ic("MidBlock")
            else:
                valid = True
        # Not in big square
        else:
            valid = False
            ic("Notin big Square")
        # Inside empty cell  
        return valid
    
    def satisfiedStateBounds(self, state: np.ndarray) -> bool:

        assert self.Umaze_x_min <= state[0] <= self.Umaze_x_max, f"X out of bound"
        assert self.Umaze_y_min <= state[1] <= self.Umaze_y_max, f"Y out of bound"
        assert -pi <= state[2] <= pi, "Yaw out of bounds "
        assert -10 <= state[3] <= 10, "x-velocity out of bounds"
        assert -10 <= state[4] <= 10, "y-velocity out of bounds"
        assert -10 <= state[5] <= 10, "yaw-velocity out of bounds"
        return True
    
    def satisfiedControlBounds(self, control: np.ndarray) -> bool:
        
        assert self.cbound_low[0] <= control[0] <= self.cbound_high[0]
        assert self.cbound_low[1] <= control[1] <= self.cbound_high[1]
        return True
