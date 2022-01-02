import argparse
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import time
from typing import Union, Tuple

import sys
import random

import gym
import d4rl

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from irl.scripts.planning.d4rlPlan.base_planner_UMaze import (
    BasePlannerUMaze,
    baseUMazeGoalState,
    baseUMazeStateValidityChecker,
)
from irl.agents.planner_utils import (
    allocateControlPlanner,
    allocateGeometricPlanner,
    make_RealVectorBounds,
)
from irl.agents.minimum_transition_objective import MinimumTransitionObjective

if __name__ == "__main__":
    env = gym.make("antmaze-umaze-v2")
    obs = env.reset()
    ic(obs)
    ic(env.spec.max_episode_steps)
    
    while 1:
        obs, *_=env.step(env.action_space.sample())
        ic(obs)
        env.render()


