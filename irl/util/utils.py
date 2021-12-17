import numpy as np
import time

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

import irl.util.pytorch_util as ptu
from irl.agents.irl_agent import IRLAgent

############################################
############################################

def sample_trajectory(
        env: gym.Env, 
        policy: Union[OffPolicyAlgorithm, IRLAgent], 
        render: Optional[bool] = False, 
        render_mode: Optional[Tuple[str]]=('rgb_array')
    ) -> Path:
    """Sample one trajectory"""
    
    # initialize env for the beginning of a new rollout
    state = env.reset() 

    # init vars
    states, acs, log_probs, rewards, next_states, terminals, image_obs = [], [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='external_camera_0', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        states.append(state.copy())
        ac, _ = policy.predict(state, deterministic=False)
        # TODO: Retrieve action log probability from policy
        log_prob = get_log_prob(policy, ac)
        
        acs.append(ac)
        log_probs.append(log_prob)

        # take that action and record results
        state, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(state.copy())
        rewards.append(rew)
        terminals.append(done)

        if done:
            break

    return Path(states, image_obs, acs, log_probs, rewards, next_states, terminals)

def get_log_prob(
        policy: OffPolicyAlgorithm,
        action: np.ndarray, 
        log_min: Optional[float] = -20.0,
        log_max: Optional[float] = 2.0,
    ) -> np.ndarray:
    """Query the SB3 policy model for log probability of action(s)"""
    if len(action.shape) > 1:
        ac = action
    else:
        ac = action[None]
    ac_tensor = ptu.from_numpy(ac)
    log_prob = policy.actor.action_dist.log_prob(ac_tensor)
    log_prob = log_prob.item()
    # Manually correct NaN values and clip range
    log_prob = log_min if np.isnan(log_prob)
    log_prob = min(log_prob, log_max)
    log_prob = max(log_prob, log_min)
    return log_prob

def sample_trajectories(
        env: gym.Env, 
        policy: Union[OffPolicyAlgorithm, IRLAgent],  
        num_trajectories: int, 
        render: Optional[bool] = False, 
        render_mode: Optional[Tuple[str]] = ('rgb_array')
    ):
    """
    Sample rollouts until we have collected num_trajectories trajectories
    """
    paths = []    
    for _ in range(num_trajectories):
        paths.append(sample_trajectory(env, policy, render, render_mode))
    return paths

def check_demo_performance(paths):
    returns = [path['reward'].sum() for path in paths]
    print(f"Collected {len(returns)} expert demonstrations")
    print(f"Demonstration return {np.mean(returns):.2f} +/ {np.std(returns):.2f}")

# def pad_absorbing_states(paths, horizon=100):
#     """
#     Add absorbing states to trajectories to make the same length
#     Add an additional dimension to normal states
#     """
#     assert len(paths) > 0, "Padding empty paths"

#     for path in paths:
#         T, state_dim = path['state'].shape
#         assert horizon >= T, "Path length larger than horizon"

#         # Add an additional dimension of zero for normal states
#         # Add absorbing states until horizon is reached
#         path['state'] = np.concatenate(
#             (path['state'], np.zeros((T,1), dtype=np.float32)),
#             axis=1)
#         absorbing_states = np.zeros((horizon-T, ob_dim+1), dtype=np.float32)
#         absorbing_states[:, -1] = 1
#         path['state'] = np.concatenate(
#             (path['state'], absorbing_states),
#             axis=0)

#         # Pad actions
#         _, ac_dim = path['action'].shape
#         path['action'] = np.concatenate(
#             (path['action'], np.zeros((horizon-T, ac_dim), dtype=np.float32)),
#             axis=0)

#         # Pad action log_probs
#         path['log_prob'] = np.concatenate(
#             (path['log_prob'], np.ones(horizon-T, dtype=np.float32)),
#             axis=0)

#         # Pad next states
#         path['next_state'] = np.concatenate(
#             (path['next_state'], np.zeros((T,1), dtype=np.float32)),
#             axis=1)
#         path['next_state'] = np.concatenate(
#             (path['next_state'], absorbing_states),
#             axis=0)

#         # Pad reward
#         path['reward'] = np.concatenate(
#             (path['reward'], np.zeros(horizon-T, dtype=np.float32)),
#             axis=0)

#         # Pad terminal
#         path['terminal'] = np.concatenate(
#             (path['terminal'], np.ones(horizon-T, dtype=np.float32)),
#             axis=0)

#     return paths


############################################
############################################

def Path(states, image_obs, acs, log_probs, rewards, next_states, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"state" : np.array(states, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "log_prob": np.array(log_probs, dtype=np.float32),
            "next_state": np.array(next_states, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

############################################
############################################

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    states = np.concatenate([path["state"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    log_probs = np.concatenate([path["log_prob"] for path in paths])
    next_states = np.concatenate([path["next_state"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return states, actions, log_probs, next_states, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean