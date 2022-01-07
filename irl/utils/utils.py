import numpy as np
import time
import matplotlib.pyplot as plt
import irl.utils.pytorch_util as ptu

############################################
############################################

def sample_trajectory(env, policy, render=False, render_mode=('rgb_array')):
    """
    Sample one trajectory 
    """
    
    # initialize env for the beginning of a new rollout
    ob = env.reset() 

    # init vars
    obs, acs, log_probs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], [], []
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
        obs.append(ob.copy())
        ac, _ = policy.predict(ob, deterministic=False)
        # TODO: Retrieve action log probability from policy
        log_prob = get_log_prob(policy, ac)
        
        acs.append(ac)
        log_probs.append(log_prob)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob.copy())
        rewards.append(rew)
        terminals.append(done)

        if done:
            break

    return Path(obs, image_obs, acs, log_probs, rewards, next_obs, terminals)

def get_log_prob(policy, action):
    """
    Query the SB3 policy model for log probability of action(s)
    """
    if len(action.shape) > 1:
        ac = action
    else:
        ac = action[None]
    ac_tensor = ptu.from_numpy(ac)
    log_prob = policy.actor.action_dist.log_prob(ac_tensor)
    log_prob = ptu.to_numpy(log_prob).item()
    # Manually correct NaN values
    if np.isnan(log_prob):
        log_prob = -20
    if log_prob < -20:
        log_prob = -20
    if log_prob > 2:
        log_prob = 2
    return log_prob

def sample_trajectories(env, policy, num_trajectories, 
                        render=False, render_mode=('rgb_array')):
    """
    Sample rollouts until we have collected num_trajectories trajectories
    """
    paths = []    
    for _ in range(num_trajectories):
        paths.append(sample_trajectory(env, policy, render, render_mode))
    return paths

def pad_absorbing_states(paths, horizon=100):
    """
    Add absorbing states to trajectories to make the same length
    Add an additional dimension to normal states
    """
    assert len(paths) > 0, "Padding empty paths"

    for path in paths:
        T, ob_dim = path['observation'].shape
        assert horizon >= T, "Path length larger than horizon"

        # Add an additional dimension of zero for normal states
        # Add absorbing states until horizon is reached
        path['observation'] = np.concatenate(
            (path['observation'], np.zeros((T,1), dtype=np.float32)),
            axis=1)
        absorbing_states = np.zeros((horizon-T, ob_dim+1), dtype=np.float32)
        absorbing_states[:, -1] = 1
        path['observation'] = np.concatenate(
            (path['observation'], absorbing_states),
            axis=0)

        # Pad actions
        _, ac_dim = path['action'].shape
        path['action'] = np.concatenate(
            (path['action'], np.zeros((horizon-T, ac_dim), dtype=np.float32)),
            axis=0)

        # Pad action log_probs
        path['log_prob'] = np.concatenate(
            (path['log_prob'], np.ones(horizon-T, dtype=np.float32)),
            axis=0)

        # Pad next observations
        path['next_observation'] = np.concatenate(
            (path['next_observation'], np.zeros((T,1), dtype=np.float32)),
            axis=1)
        path['next_observation'] = np.concatenate(
            (path['next_observation'], absorbing_states),
            axis=0)

        # Pad reward
        path['reward'] = np.concatenate(
            (path['reward'], np.zeros(horizon-T, dtype=np.float32)),
            axis=0)

        # Pad terminal
        path['terminal'] = np.concatenate(
            (path['terminal'], np.ones(horizon-T, dtype=np.float32)),
            axis=0)

    return paths


def visualize_trajectory(trajectory):
    x, y = trajectory.states[:, 0], trajectory.states[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    plt.show()

def render_trajectory(env, qpos, qvel):    
    for i in range(qpos.shape[0]):
        env.set_state(qpos[i], qvel[i])
        env.render()