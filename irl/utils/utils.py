from typing import Optional, Union, Any, Tuple, List

import gym
import numpy as np
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from irl.utils import pytorch_utils as ptu
from irl.utils import types

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
    obs, acs, log_probs, rewards, next_obs, terminals = [], [], [], [], [], []
    env.reset()
    ob, rew, done, info = env.step(env.action_space.sample())
    infos = {}
    for key in info.keys():
        infos[key] = []

    # initialize env for the beginning of a new rollout
    # reset to given mujoco state is qpos, qvel are provided
    ob = env.reset() 
    if qpos is not None and qvel is not None:
        env.set_state(qpos, qvel)

    while True:
        # use the most recent ob to decide what to do
        obs.append(ob.copy())
        # ac, _ = policy.predict(ob, deterministic=False)        
        ac, log_prob = action_log_prob(policy, ob)
        acs.append(ac)
        log_probs.append(log_prob)

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
        infos=infos,
        log_probs=np.array(log_probs)
    )

def action_log_prob(
    policy: Union[OffPolicyAlgorithm, Actor],
    ob: np.ndarray,
    log_min: Optional[float] = -20.0,
    log_max: Optional[float] = 2.0,
) -> np.ndarray:
    """Query SB3 policy model for action and corresponding log probability"""
    if isinstance(policy, OffPolicyAlgorithm):
        policy = policy.policy.actor
    elif isinstance(policy, Actor):
        pass
    else:
        raise ValueError(f"Policy type {type(policy)} is not implemented")
    ob, _ = policy.obs_to_tensor(ob)
    action, log_prob = policy.action_log_prob(ob)
    action = ptu.to_numpy(action)[0]
    log_prob = ptu.to_numpy(log_prob)[0]
    log_prob = np.clip(log_prob, log_min, log_max)
    assert log_min <= log_prob <= log_max, f"log_prob {log_prob:.2f} not in bounds"
    return action, log_prob

def sample_agent_action_log_prob(
    demo_states: th.Tensor,
    policy: OffPolicyAlgorithm,
) -> Tuple[th.Tensor, th.Tensor]:
    """Sample agent actions and log probabilities from demo states"""
    if isinstance(policy, SAC):
        policy = policy.policy.actor
    elif isinstance(policy, Actor): 
        policy = policy
    else:
        assert False, f"Policy class {type(policy)} not implemented"
    actions, log_probs = policy.action_log_prob(demo_states)
    return actions, log_probs

# def eval_log_probs(
#     policy: Union[OffPolicyAlgorithm, Actor],
#     ob: th.Tensor,
#     ac: th.Tensor,
#     log_min: Optional[float] = -20.0,
#     log_max: Optional[float] = 2.0,
# ) -> np.ndarray:
#     """Query SB3 policy model for action log probability"""
#     if isinstance(policy, OffPolicyAlgorithm):
#         policy = policy.policy.actor
#     elif isinstance(policy, Actor):
#         pass
#     else:
#         raise ValueError(f"Policy type {type(policy)} is not implemented")
#     # ob, _ = policy.obs_to_tensor(ob)
#     batch_size, T, ob_dim = ob.shape
#     ac_dim = ac.shape[-1]
#     ob = ob.reshape(-1, ob_dim)
#     ac = ac.reshape(-1, ac_dim)

#     mean_actions, log_std, kwargs = policy.get_action_dist_params(ob)
#     policy.action_dist.proba_distribution(mean_actions, log_std)
#     log_probs = policy.action_dist.log_prob(ac)
#     log_probs = log_probs.reshape(batch_size, T)
#     return log_probs

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

    vel_max = 50
    assert ((states_min[2:4] >= np.array([-vel_max, -vel_max])).all() and 
        (states_max[2:4] <= np.array([vel_max, vel_max])).all()), (
        f"Demonstrations angular velocity not in range {[-vel_max, vel_max]}, need to rerun collection"
    ) 

def collect_demo_trajectories(env: gym.Env, expert_policy: str, batch_size: int):
    expert_policy = SAC.load(expert_policy)
    print('\nRunning expert policy to collect demonstrations...')
    demo_paths = sample_trajectories(env, expert_policy, batch_size)
    check_demo_performance(demo_paths)
    return demo_paths

def extract_paths(paths: List[types.Trajectory]) -> List[th.Tensor]:
    obs = ptu.from_numpy(np.array([path.states for path in paths]))
    # Drop the last terminal state
    obs = obs[:, :-1, :]
    act = ptu.from_numpy(np.array([path.actions for path in paths]))
    log_probs = ptu.from_numpy(np.array([path.log_probs for path in paths]))
    assert obs.shape[0] == act.shape[0] == log_probs.shape[0], (
        "Batch size is not same for extracted paths"
    )
    assert obs.shape[1] == act.shape[1] == log_probs.shape[1], (
        "Episode length is not same for extracted paths"
    )
    return obs, act, log_probs

def log_disc_metrics(logger, metrics):
    for k, v in metrics.items():
        if v.dim() < 1 or (v.dim() == 1 and v.shape[0] <= 1):
            logger.record_mean(f"Reward/{k}", v.item())
        else:
            logger.record_mean(f"Reward/{k}Max", th.amax(v).item())
            logger.record_mean(f"Reward/{k}Min", th.amin(v).item())
            logger.record_mean(f"Reward/{k}Mean", th.mean(v).item())
            logger.record_mean(f"Reward/{k}Std", th.std(v).item())