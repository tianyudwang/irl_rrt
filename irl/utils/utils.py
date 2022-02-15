from typing import Optional, Union, Any, Tuple, List
import time

import gym
import numpy as np

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from irl.utils import pytorch_util as ptu
from irl.utils import types

############################################
############################################

def sample_trajectory(
    env: gym.Env, 
    policy: OffPolicyAlgorithm, 
    qpos: Optional[np.ndarray] = None, 
    qvel: Optional[np.ndarray] = None
) -> types.TrajectoryWithReward:
    """Sample one trajectory with policy. 
    Set mujoco state if qpos and qvel are provided
    """

    # init vars
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    env.reset()
    ob, rew, done, info = env.step(env.action_space.sample())
    infos = {}
    for key in info.keys():
        infos[key] = []

    # initialize env for the beginning of a new rollout
    ob = env.reset() 
    if qpos is not None and qvel is not None:
        env.set_state(qpos, qvel)

    while True:
        # use the most recent ob to decide what to do
        obs.append(ob.copy())
        ac, _ = policy.predict(ob, deterministic=False)        
        acs.append(ac)
        # log_probs.append(log_prob)

        # take that action and record results
        ob, rew, done, info = env.step(ac)

        # record result of taking that action
        next_obs.append(ob.copy())
        rewards.append(rew)
        terminals.append(done)
        for key, val in info.items():
            infos[key].append(val)

        if done:
            obs.append(ob.copy())
            break

    for key, val in infos.items():
        infos[key] = np.array(val)

    return types.TrajectoryWithReward(
        states=np.array(obs), 
        actions=np.array(acs), 
        rewards=np.array(rewards),
        infos=infos
    )


def action_log_prob(
    policy: OffPolicyAlgorithm,
    ob: np.ndarray,
    log_min: Optional[float] = -20.0,
    log_max: Optional[float] = 2.0,
) -> np.ndarray:
    """Query SB3 policy model for action and corresponding log probability"""
    assert isinstance(policy, OffPolicyAlgorithm), (
        f"Policy type {type(policy)} is not OffPolicyAlgorithm"
    )
    ob, _ = policy.policy.obs_to_tensor(ob)
    action, log_prob = policy.actor.action_log_prob(ob)
    action = ptu.to_numpy(action)[0]
    log_prob = ptu.to_numpy(log_prob)[0]
    log_prob = np.clip(log_prob, log_min, log_max)
    assert log_min <= log_prob <= log_max, f"log_prob {log_prob:.2f} not in bounds"
    return action, log_prob


def sample_trajectories(
    env: gym.Env, 
    policy: Union[OffPolicyAlgorithm, Any],  
    num_trajectories: int
):
    """Sample rollouts until we have collected num_trajectories trajectories"""
    paths = []    
    for _ in range(num_trajectories):
        paths.append(sample_trajectory(env, policy))
    return paths

def check_demo_performance(paths):
    assert type(paths[0]) == types.TrajectoryWithReward, "Demo path type is not types.TrajectoryWithReward"
    returns = [path.rewards.sum() for path in paths]
    lens = [len(path) for path in paths]
    print(f"Collected {len(returns)} expert demonstrations")
    print(f"Demonstration length {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"Demonstration return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")

    # For Reacher-v2, check angular velocity is in range [-10, 10]
    states = np.concatenate([path.states for path in paths], axis=0)
    states_min = np.amin(states, axis=0)
    states_max = np.amax(states, axis=0)

    assert ((states_min[2:4] >= np.array([-10, -10])).all() and 
        (states_max[2:4] <= np.array([10, 10])).all()), (
        "Demonstrations angular velocity not in range [-10, 10], need to rerun collection"
    ) 


