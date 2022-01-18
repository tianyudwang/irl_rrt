"""
Usage: pytest test_antmaze.py or pytest test_antmaze.py::test_feasible_region
Check antmaze-umaze-v1 environment specifications
"""

import gym 
import d4rl
import numpy as np 
import time

from irl.planners.geometric_planner import AntMazeRRTstarPlanner
# from irl.planners.control_planner import AntMazeSSTPlanner, AntMazeKPIECEPlanner
from irl.utils import wrappers

def test_antmaze_wrappers():
    """
    Test one step transition function 
    """
    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)

    env = wrappers.AntMazeFixedGoalWrapper(env)
    env = wrappers.AntMazeFixedStartWrapper(env)
    env = wrappers.AntMazeTransitionWrapper(env)

    state = env.reset()
    path_1, actions = [state], []

    for i in range(10):
        action = env.action_space.sample()
        state, _, _, _ = env.step(action)
        path_1.append(state)
        actions.append(action) 


    state = env.reset()
    path_2 = [state]
    for action in actions:
        state = env.one_step_transition(state, action)
        path_2.append(state)

    for state_1, state_2 in zip(path_1, path_2):
        assert np.allclose(state_1, state_2)


def test_feasible_region():
    """
    The antmaze-umaze-v1 specification is 
    U_MAZE_TEST = [[1, 1, 1, 1, 1],
                   [1, R, 0, 0, 1],
                   [1, 1, 1, 0, 1],
                   [1, G, 0, 0, 1],
                   [1, 1, 1, 1, 1]]
    The scaling is 4 and reset location is (0, 0) after offset
    The empty region is the square [-2, 10] x [-2, 10], 
    excluding the rectangle [-2, 6] x [2, 6]
    The ant size is 1 assuming legs are stretched straight, 
    thus the feasible area for ant center is 
    [-1, 9] x [-1, 9], excluding [-1, 7] x [1, 7]
    """
    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)

    square_low = np.array([-1, -1])
    square_high = np.array([9, 9])
    rectangle_low = np.array([-1, 1])
    rectangle_high = np.array([7, 7])

    qpos, qvel = np.zeros(15), np.zeros(14)
    qpos[2] = 1
    qpos[3] = 1

    for i in range(10000):
        loc = np.random.uniform(low=square_low, high=square_high)

        # Skip locations within the rectangle
        if (rectangle_low <= loc).all() and (loc <= rectangle_high).all():
            continue 
        qpos[:2] = loc
        env.set_state(qpos, qvel)

        dist = np.amin([contact.dist for contact in env.data.contact])
        assert dist >= 0, f"{loc} is in collision with distance {dist}"

def test_antmaze_RRTstar_planner():

    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)
    goal = (0, 8)
    env.set_target_goal(goal)

    planner = AntMazeRRTstarPlanner()

    for _ in range(10):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs, solveTime=10)
        # import ipdb; ipdb.set_trace()

        # for state in states:
        #     qpos, qvel = state[:15], state[15:]
        #     env.set_state(qpos, qvel)
        #     env.render()
        #     time.sleep(1)
        assert np.linalg.norm([states[-1][0] - goal[0], states[-1][1] - goal[1]]) <= 0.5

# def test_antmaze_SST_planner():
#     env_name = "antmaze-umaze-v1"
#     env = gym.make(env_name)
#     goal = (0, 8)
#     env.set_target_goal(goal)

#     planner = AntMazeSSTPlanner(env)

#     for _ in range(1):
#         obs = env.reset()
#         status, states, controls = planner.plan(start=obs, solveTime=60)

#         print(states[:, :2])

#         for state in states:
#             qpos, qvel = state[:15], state[15:]
#             env.set_state(qpos, qvel)
#             env.render()
#             time.sleep(0.1)

#         import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    test_antmaze_wrappers()
    # test_feasible_region()
    # test_antmaze_RRTstar_planner()
    # test_antmaze_SST_planner()
    # test_antmaze_KPIECE_planner()