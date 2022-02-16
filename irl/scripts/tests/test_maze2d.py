"""
Usage: pytest test_maze2d.py or pytest test_maze2d.py::test_fixed_start
Check maze2d-umaze-v1 environment specifications
"""

import gym
import d4rl
import numpy as np

import irl.planners.geometric_planner as gp 
import irl.planners.control_planner as cp
import irl.utils.wrappers as wrappers
import irl.planners.planner_utils as planner_utils

def test_fixed_start():
    """Test whether reset location is at (3, 1)"""
    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedStartWrapper(env)

    start_loc = np.array([3., 1.])

    for i in range(100):
        state = env.reset()
        qpos = state[:2]
        assert np.linalg.norm(qpos - start_loc) <= 0.2

def test_feasible_region():
    """
    The umaze string spec is 
    U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"
    with start at (3, 1) and goal at (1, 1)
    The empty region is the square [0.5, 3.5] x [0.5, 3.5], 
    excluding the rectangle [1.5, 2.5] x [0.5, 2.5]
    The point size is 0.1, thus the feasible area for point center is 
    [0.6, 3.4] x [0.6, 3.4], excluding [1.4, 2.6] x [0.6, 2.6]
    """
    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedLocationWrapper(env)

    square_low = np.array([0.6, 0.6])
    square_high = np.array([3.4, 3.4])
    rectangle_low = np.array([1.4, 0.6])
    rectangle_high = np.array([2.6, 2.6])

    for i in range(10000):
        loc = np.random.uniform(low=square_low, high=square_high)

        # Skip locations within the rectangle
        if (rectangle_low <= loc).all() and (loc <= rectangle_high).all():
            continue

        # Simulate the location and check for collision
        env.reset_to_location(loc)
        
        # Ball is always in contact with ground
        # Negative contact distance means collision
        for contact in env.data.contact:
            assert contact.dist >= 0

def test_maze2d_RRTstar_planner():
    """
    Reset the maze2d-umaze-v1 environment and use initial state
    to plan optimal path to goal at (1, 1)
    """

    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedStartWrapper(env)
    goal = np.array([1., 1.])

    planner = gp.Maze2DRRTstarPlanner()
    for _ in range(10):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs)
        assert np.linalg.norm([states[-1][0] - goal[0], states[-1][1] - goal[1]]) <= 0.1, (
            f"Last state {states[-1]} does not reach goal at {goal}"
        )

        # planner_utils.visualize_path(states, goal, save=True)

def test_maze2d_PRMstar_planner():
    """
    Reset the maze2d-umaze-v1 environment and use initial state
    to plan optimal path to goal at (1, 1)
    """

    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedStartWrapper(env)
    goal = np.array([1., 1.])

    planner = gp.Maze2DPRMstarPlanner()

    for _ in range(10):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs)
        assert np.linalg.norm([states[-1][0] - goal[0], states[-1][1] - goal[1]]) <= 0.1

        # planner_utils.visualize_path(states, goal, save=True)

# def test_maze2d_PRMstar_multiple_queries():
#     """
#     Random start locations with multiple queries 
#     Reuses previous planning data
#     This function is not working yet. 
#     Planning function takes significantly long time and can get stuck.
#     """

#     env_name = "maze2d-umaze-v1"
#     env = gym.make(env_name)

#     goal = np.array([1., 1.])

#     planner = Maze2DPRMstarPlanner()

#     for i in range(10):
#         obs = env.reset()
#         print(obs)

#         status, states, controls = planner.plan_exact_solution(start=obs, clear_query=True)
#         assert np.linalg.norm([states[-1][0] - goal[0], states[-1][1] - goal[1]]) <= 0.1



def test_maze2d_SST_planner():
    """
    Reset the maze2d-umaze-v1 environment and use initial state
    to plan optimal path to goal at (1, 1)
    """

    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedStartWrapper(env)
    goal = np.array([1., 1.])

    planner = cp.Maze2DSSTPlanner(env.unwrapped)
    for _ in range(10):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs)

        # planner_utils.visualize_path(states, goal, save=True)
        # Check rollout states from controls and compare to planned states
        obs = env.reset()
        # env.render()
        rollout_states = [obs]
        for j in range(len(controls)):
            control = controls[j]
            obs, _, _, _ = env.step(control)
            # env.render()
            rollout_states.append(obs)
        rollout_states = np.array(rollout_states)

        assert len(rollout_states) == len(states)
        assert np.linalg.norm(rollout_states - states) < 0.1

def test_maze2d_KPIECE_planner():
    """
    Reset the maze2d-umaze-v1 environment and use initial state
    to plan optimal path to goal at (1, 1)
    """

    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedStartWrapper(env)
    goal = np.array([1., 1.])

    planner = cp.Maze2DKPIECEPlanner(env.unwrapped)
    for _ in range(1):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs)

        print(states[:, :2])
        # planner_utils.visualize_path(states, goal, save=True)
        # Check rollout states from controls and compare to planned states
        obs = env.reset()
        # env.render()
        rollout_states = [obs]
        for j in range(len(controls)):
            control = controls[j]
            obs, _, _, _ = env.step(control)
            # env.render()
            rollout_states.append(obs)
        rollout_states = np.array(rollout_states)

        assert len(rollout_states) == len(states)
        assert np.linalg.norm(rollout_states - states) < 0.1

def test_maze2d_RRT_planner():
    """
    Reset the maze2d-umaze-v1 environment and use initial state
    to plan optimal path to goal at (1, 1)
    """

    env_name = "maze2d-umaze-v1"
    env = gym.make(env_name)
    env = wrappers.Maze2DFixedStartWrapper(env)
    goal = np.array([1., 1.])

    planner = cp.Maze2DRRTPlanner(env.unwrapped)
    for _ in range(10):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs, solveTime=20)

        # planner_utils.visualize_path(states, goal, save=True)

        # Check rollout states from controls and compare to planned states
        obs = env.reset()
        rollout_states = [obs]
        for j in range(len(controls)):
            control = controls[j]
            obs, _, _, _ = env.step(control)
            rollout_states.append(obs)
        rollout_states = np.array(rollout_states)

        assert len(rollout_states) == len(states)
        assert np.linalg.norm(rollout_states - states) < 0.1

if __name__ == '__main__':
    # test_feasible_region()
    # test_maze2d_RRTstar_planner()
    # test_maze2d_PRMstar_multiple_queries()
    # test_maze2d_PRMstar_planner()
    # test_maze2d_RRT_planner()
    # test_maze2d_SST_planner()
    test_maze2d_KPIECE_planner()