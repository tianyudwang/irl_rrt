import multiprocessing as mp
import time

import gym
import d4rl
import numpy as np

import irl.planners.geometric_planner as gp
import irl.utils.wrappers as wrappers

def plan_start(start):
    planner = gp.Maze2DRRTstarPlanner()
    status, states, controls = planner.plan(start)
    return states

def test_multiprocessing():
    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedStartWrapper(env)
    obs = env.reset()

    start_states = [obs for _ in range(8)] 

    start_time = time.time()
    with mp.Pool() as pool:
        paths = pool.map(plan_start, start_states)

    end_time = time.time() - start_time
    print(f"Planning {len(start_states)} starts in {end_time:.2f}s")
    print(f"Path lengths are {[len(path) for path in paths]}")
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    test_multiprocessing()
