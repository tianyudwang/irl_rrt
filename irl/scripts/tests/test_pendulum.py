import gym
import numpy as np

import irl.planners.geometric_planner as gp
import irl.planners.control_planner as cp
from irl.utils.wrappers import PendulumWrapper
from irl.utils import planner_utils

def init_env():
    seed = 0
    rng = np.random.RandomState(seed)
    env_seed = rng.randint(0, (1 << 31) - 1)
    env = PendulumWrapper(gym.make("Pendulum-v1"))
    env.seed(int(env_seed))
    return env

def test_one_step_transition():
    env = init_env()

    for i in range(100):
        state = env.reset()
        path_1, path_2, actions = [state], [state], []
        for _ in range(10):
            action = env.action_space.sample()
            state, _, _, _ = env.step(action)
            path_1.append(state)
            actions.append(action)  

        state = path_2[0]
        for action in actions:  
            state = env.one_step_transition(state, action)
            path_2.append(state)    

        for state_1, state_2 in zip(path_1, path_2):
            assert np.allclose(state_1, state_2), (
                f"state 1 {state_1} and state 2 {state_2} does not match"
            )

def test_reset():
    env = init_env()
    high = np.array([np.pi, 8])

    for _ in range(100):
        reset_location = np.random.uniform(low=-high, high=high)
        obs = env.reset(reset_location)
        assert np.allclose(obs, reset_location)

def test_rrtstar():
    env = init_env()
    planner = gp.PendulumRRTstarPlanner()

    for _ in range(10):
        obs = env.reset()

        status, states, _ = planner.plan(obs, solveTime=0.4)

        assert np.linalg.norm(states[-1]) <= 0.1, (
            f"Final state {states[-1]} does not reach goal"
        )
        # planner_utils.visualize_path(states)

def test_sst():

    env = init_env()
    planner = cp.PendulumSSTPlanner()
    goal = np.array([0., 0.])

    for _ in range(10):
        obs = env.reset()

        status, states, controls = planner.plan(obs, solveTime=1.0)

        assert np.linalg.norm(states[-1]) <= 0.1, (
            f"Final state {states[-1]} does not reach goal"
        )
        # planner_utils.visualize_path(states)

 
if __name__ == '__main__':
    # test_one_step_transition()
    # test_rrtstar()
    test_sst()
