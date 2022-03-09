import time
import gym 
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from ompl import base as ob
from ompl import control as oc

from irl.utils.wrappers import IRLEnv, ReacherWrapper
from irl.utils import planner_utils
from irl.rewards.reward_net import RewardNet
from irl.planners import base_planner as bp
from irl.planners import geometric_planner as gp 
from irl.planners import control_planner as cp


np.set_printoptions(precision=6, suppress=True)

def test_compute_xy_from_angles():
    env_name = "Reacher-v2"
    env = gym.make(env_name)
    env = ReacherWrapper(env)

    for i in range(10000):
        ob = env.reset(random=True)

        th1, th2 = ob[0], ob[1]
        xy = planner_utils.compute_xy_from_angles(th1, th2)
        xy = np.array(xy)

        assert np.allclose(ob[4:6], xy, atol=1e-6), (
            f"true fingertip {ob[4:6]}, computed fingertip {xy}"
        )


def test_compute_angles_from_xy():
    """
    planner_utils.compute_angles_from_xy has numerical precision issues
    angles do not recover the fingertip position precisely
    """
    env_name = "Reacher-v2"
    env = gym.make(env_name)
    env = ReacherWrapper(env)

    for i in range(1000):
        ob = env.reset(random=True)
        # print(f"arm1 start loc {env.unwrapped.get_body_com('body1')[:2]}")
        th1, th2 = planner_utils.compute_angles_from_xy(ob[4], ob[5])   

        # convert angles back to xy and should match fingertip
        xy = planner_utils.compute_xy_from_angles(th1, th2)
        xy = np.array(xy)

        dist = np.linalg.norm(xy - ob[4:6])
        assert dist < 1e-2, (
            f"true fingertip {ob[4:6]}, computed fingertip {xy}"
        )

 
def test_reacher_next_state():
    env = gym.make("Reacher-v2")
    env = ReacherWrapper(env)

    # Check if next state is the same given the same current state and action
    # MUST copy qpos and qvel
    env.reset()
    qpos = env.unwrapped.sim.data.qpos.ravel()[:].copy()
    qvel = env.unwrapped.sim.data.qvel.ravel()[:].copy()

    action = env.action_space.sample()
    states_before, states_after = [], []
    n_states = 1000
    
    for i in range(n_states):
        env.reset()
        env.set_state(qpos, qvel)
        states_before.append(env._get_obs())
        states_after.append(env.step(action)[0])

    states_before = np.stack(states_before)
    states_after = np.stack(states_after)
    states_before_diff = np.linalg.norm(states_before - np.mean(states_before, axis=0)) / n_states
    states_after_diff = np.linalg.norm(states_after - np.mean(states_after, axis=0)) / n_states

    assert states_before_diff < 1e-6, (
        f"State difference: {states_before_diff:.8f}")
    assert states_after_diff < 1e-6, (
        f"Next state difference: {states_after_diff:.8f}")

# def test_reacher_StatePropagator():
#     env = gym.make("Reacher-v2")
#     env = ReacherWrapper(env)

#     # Sample random state and next state from env and compare to ompl propagate function
#     planner = cp.ReacherSSTPlanner(env.unwrapped)
#     state_propagator = planner.state_propagator

#     for i in range(1000):
#         # gym step
#         obs = env.reset()
#         qpos = env.unwrapped.sim.data.qpos.ravel()[:].copy()
#         qvel = env.unwrapped.sim.data.qvel.ravel()[:].copy()
#         gym_state_before = obs[:6].astype(np.float64)
#         action = env.action_space.sample()
#         obs, rew, done, info = env.step(action)
#         gym_state_after = obs[:6]

#         # ompl propagate
#         state_before = planner.get_StartState(gym_state_before)
#         state_after = ob.State(planner.space)
#         control = planner.cspace.allocControl()
#         action = action.astype(np.float64)
#         control[0] = action[0]
#         control[1] = action[1]
#         state_propagator.propagate(state_before(), control, 2.0, state_after()) 
#         ompl_state_after = planner_utils.convert_ompl_state_to_numpy(state_after())

#         # compare
#         assert np.allclose(ompl_state_after, gym_state_after, atol=1e-6), (
#             f"States do not match after ompl propagte \n"
#             f"obs before {gym_state_before} \n"
#             f"qpos before {qpos} \n"
#             f"qvel before {qvel} \n"
#             f"action {action} \n"
#             f"gym state after {gym_state_after} \n"
#             f"ompl state after {ompl_state_after} \n"
#         )

def reacher_cost_fn(s1, s2):
    assert (s1[-2:] == s2[-2:]).all(), "Target state not equal"
    cost = np.linalg.norm(s1[:-2] - s2[:-2]) + np.linalg.norm(s2[4:6] - s2[6:8])
    return cost

def test_reacher_RRTstar_planner():

    env_name = "Reacher-v2"
    env = gym.make(env_name)
    env = ReacherWrapper(env)

    planner = gp.ReacherRRTstarPlanner()
    for _ in range(100):
        obs = env.reset()
        start = obs[:4].astype(np.float64)
        target = obs[-2:].astype(np.float64)
        planner.update_ss_cost(reacher_cost_fn, target)
        status, states, controls = planner.plan(start=start, goal=target)

        # env.reset()
        # for state in states:
        #     qpos, qvel = np.zeros(4), np.zeros(4)
        #     qpos[:2] = state[:2]
        #     qvel[:2] = state[2:4]
        #     qpos[2:4] = target
        #     env.set_state(qpos, qvel)
        #     env.render()
        #     time.sleep(0.01)

        finger_pos = planner_utils.compute_xy_from_angles(states[-1][0], states[-1][1])
        dist = np.linalg.norm(target - finger_pos)
        assert dist <= 0.01, (
            f"Reacher finger position {states[-1]} does not reach target at {target}",
            f"Distance to target is {dist}"
        )

def test_reacher_PRMstar_planner():
    """
    OMPL PRM style planners do not support multiple queries. 
    clearQuery() function does not clear start and goal from previous planning.
    """
    env_name = "Reacher-v2"
    env = gym.make(env_name)
    env = ReacherWrapper(env)

    planner = gp.ReacherPRMstarPlanner()
    for _ in range(10):
        obs = env.reset()
        start = obs[:4].astype(np.float64)
        target = obs[-2:].astype(np.float64)
        # target = np.array([-0.21, 0], dtype=np.float64)
        status, states, controls = planner.plan(start=start, goal=target)

        print(start, states[0])
        print(f"Start state {start}")
        print(f"First state from planner {states[0]}")
        print(f"Start state distance {np.linalg.norm(states[0] - start):.2f}")
        print(f"Last state from planner {states[-1]}")
        print(f"Target location {target}")

        # print(states)
        finger_pos = planner_utils.compute_xy_from_angles(states[-1][0], states[-1][1])
        dist = np.linalg.norm(target - finger_pos)
        print(f"Final state fingertip dist {dist:.2f}")

        env.reset()
        print(len(states))
        for state in states:
            print(state)
            qpos, qvel = np.zeros(4), np.zeros(4)
            qpos[:2] = state[:2]
            qvel[:2] = state[2:4]
            qpos[2:4] = target
            env.set_state(qpos, qvel)
            env.render()
            time.sleep(0.01)

        if dist >= 0.02:
            import ipdb; ipdb.set_trace()
        assert dist <= 0.05, (
            f"Reacher finger position {states[-1]} does not reach target at {target}",
            f"Distance to target is {dist}"
        )

# def test_reacher_SST_planner():
#     """
#     Path returned by planner does not match exactly the rollout path from controls
#     Major discrepancy in angle and angular velocity in joint1 (the second arm)
#     """
#     assert False, "This test function does not work"
#     env_name = "Reacher-v2"
#     env = gym.make(env_name)
#     env = ReacherWrapper(env)

#     planner = cp.ReacherSSTPlanner(env.unwrapped)
#     for _ in range(10):
#         obs = env.reset()
#         qpos = env.unwrapped.sim.data.qpos.flat[:].copy()
#         qvel = env.unwrapped.sim.data.qvel.flat[:].copy()
#         start = obs[:-2].astype(np.float64)
#         target = obs[-2:].astype(np.float64)
#         status, states, controls = planner.plan(start=start, goal=target)

#         finger_pos = states[-1][-2:]
#         dist = np.linalg.norm(target - finger_pos)
#         assert dist <= 0.01, (
#             f"Reacher state {states[-1]} does not reach target at {target} \n"
#             f"Distance to target is {dist}"
#         )

#         env.reset()
#         env.set_state(qpos, qvel)
#         new_obs = env._get_obs()
#         assert np.allclose(obs, new_obs)

#         # env.render()
#         rollout_states = [obs]
#         for j in range(len(controls)):
#             control = controls[j]
#             obs, _, _, _ = env.step(control)
#             # env.render()
#             rollout_states.append(obs)
#         rollout_states = np.array(rollout_states)

#         assert len(rollout_states) == len(states)
#         if np.linalg.norm(rollout_states[:, :6] - states) >= 0.1:   
#             import ipdb; ipdb.set_trace()
#         # assert np.linalg.norm(rollout_states[:, :6] - states) < 0.1


# def eval(env, model):
#     rews = []
#     n_eval = 64
#     for i in range(n_eval):
#         obs = env.reset()
#         done = False
#         rewards = []
#         while not done:
#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
#             rewards.append(reward)
#         rews.append(rewards)    

#     lengths = [len(rew) for rew in rews]
#     returns = [sum(rew) for rew in rews]
#     print(f"Reacher-v2 {n_eval} episodes")
#     print(f"Episode return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
#     print(f"Episode length {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")

if __name__ == '__main__':
    # test_compute_xy_from_angles()
    # test_compute_angles_from_xy()
    # test_reacher_fingertip()
    # test_reacher_RRTstar_planner()
    test_reacher_PRMstar_planner()
    # test_reacher_SST_planner()
    # test_reacher_StatePropagator()
