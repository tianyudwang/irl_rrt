import pickle

import gym
import mujoco_maze
import numpy as np
from icecream import ic

from irl.scripts.wrapper.remove_timeDim_wrapper import RemovTimeFeatureWrapper
from irl.scripts.wrapper.one_step_PointUMaze_wrapper import PointUMazeOneStepTransitionWrapper


class DummyStartStateValidityChecker:
    def __init__(
        self,
    ):
        # Point radius
        self.size = 0.5
        self.Umaze_x_min = self.Umaze_y_min = -2 + self.size
        self.Umaze_x_max = self.Umaze_y_max = 10 - self.size

    def isValid(self, state: np.ndarray) -> bool:

        #
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
            inMidBlock = (-2 <= x_pos <= 6.5) and (1.5 <= y_pos <= 6.5)
            if inMidBlock:
                valid = False
            else:
                valid = True
        # Not in big square
        else:
            valid = False

        # Inside empty cell and satisfiedBounds
        return valid

if __name__ == "__main__":

    env = gym.make("PointUMaze-v0")
    env = PointUMazeOneStepTransitionWrapper(RemovTimeFeatureWrapper(env))

    with open("./data.pkl", "rb") as f:
        data = pickle.load(f)
    ic(data)

    validationChecker = DummyStartStateValidityChecker()
    ic(validationChecker.isValid(data["ob"]))
