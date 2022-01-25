"""
Usage: pytest test_antmaze.py or pytest test_antmaze.py::test_feasible_region
Check antmaze-umaze-v1 environment specifications
"""

import gym 
import d4rl
import numpy as np 
import time

import irl.planners.geometric_planner as gp
import irl.planners.control_planner as cp
import irl.planners.base_planner as bp
from irl.utils import wrappers

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

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

def test_antmaze_start():
    """
    Check if dataset states satisfy state space bounds
    OMPL complains state does not satisfy bounds but after enforceBounds 
    the state remains the same
    """
    import h5py
    from ompl import base as ob 

    dataset = h5py.File('../antmaze_umaze_fixstart_fixgoal.hdf5', 'r')
    assert dataset['observations'].shape[1] == 29

    states = dataset['observations']

    planner = gp.AntMazeBasePlanner()
    space = planner.get_StateSpace()

    for state in states:
        start_state = ob.State(space)

        for i in range(len(state)):
            start_state[i] = state[i].item()

        if not space.satisfiesBounds(start_state()):
            space.enforceBounds(start_state())

        for i in range(len(state)):
            assert np.abs(start_state[i] - state[i]) < 1e-3, f"{i}, {state[i]} , {start_state[i]}"



def test_antmaze_RRTstar_planner():

    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)
    goal = (0, 8)
    env.set_target_goal(goal)

    planner = gp.AntMazeRRTstarPlanner()

    for _ in range(10):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs)
        # import ipdb; ipdb.set_trace()

        for state in states:
            qpos, qvel = state[:15], state[15:]
            env.set_state(qpos, qvel)
            env.render()
            time.sleep(1)
        assert np.linalg.norm([states[-1][0] - goal[0], states[-1][1] - goal[1]]) <= 0.5




def test_antmaze_SST_planner():
    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)
    goal = (0, 8)
    env.set_target_goal(goal)

    planner = cp.AntMazeSSTPlanner(env)

    for _ in range(1):
        obs = env.reset()
        status, states, controls = planner.plan(start=obs)

        print(states[:, :2])

        for state in states:
            qpos, qvel = state[:15], state[15:]
            env.set_state(qpos, qvel)
            env.render()
            time.sleep(0.1)

        import ipdb; ipdb.set_trace()


def test_antmaze_KPIECE_planner():
    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)
    goal = (0, 8)
    env.set_target_goal(goal)

    planner = cp.AntMazeKPIECEPlanner(env)

    for _ in range(1):
        obs = env.reset()
        status, states, controls = planner.plan(
            start=obs, 
            solveTime=10,
            total_solveTime=10000
        )

        print(states[:, :2])

        np.save('states', states)
        np.save('controls', controls)
        import ipdb; ipdb.set_trace()

        for state in states:
            qpos, qvel = state[:15], state[15:]
            env.set_state(qpos, qvel)
            env.render()
            time.sleep(0.1)

        import ipdb; ipdb.set_trace()

def load_path():
    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)
    goal = (0, 8)
    env.set_target_goal(goal)

    states = np.load('states.npy')
    controls = np.load('controls.npy')

    obs = env.reset()
    env.render()

    for state in states:
        qpos, qvel = state[:15], state[15:]
        env.set_state(qpos, qvel)
        env.render()

    # for control in controls:
    #     obs, _, _, _ = env.step(control)
    #     env.render()

def sac_learn_ant():
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure

    env_name = "antmaze-umaze-v1"
    env = gym.make(env_name)
    import ipdb; ipdb.set_trace()
    # env = wrappers.AntMazeFixedGoalWrapper(env)
    # env = wrappers.AntMazeFixedStartWrapper(env, start=(2, 8))
    # env = wrappers.AntMazeFirstExitWrapper(env)
    # env = gym.make("Ant-v2")

    tmp_path = "/tmp/sac_ant_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    policy_kwargs = dict(net_arch=[256, 256])
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=int(1e7), log_interval=4)
    model.save("sac_ant")







if __name__ == '__main__':
    # test_antmaze_wrappers()
    # test_feasible_region()
    # test_antmaze_start()
    # test_antmaze_RRTstar_planner()
    # test_antmaze_SST_planner()
    # test_antmaze_KPIECE_planner()
    # load_path()
    sac_learn_ant()