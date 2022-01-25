import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import numpy as np
import pickle
import gzip
import h5py
import argparse
from d4rl.locomotion import maze_env, ant
from d4rl.locomotion.wrappers import NormalizedBoxEnv
import torch
from PIL import Image

from tqdm import tqdm
from icecream import ic

from irl.scripts.wrapper.antWrapper import (
    AntMazeFixedStartWrapper,
    AntMazeFixedGoalWrapper,
    AntMazeFixStartAndGoalWrapper
)


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, r, tgt, done, timeout, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def extend_data(src, target):
    src['observations'].extend(target['observations'])
    src['actions'].extend(target['actions']     )
    src['rewards'].extend(target['rewards']     )
    src['terminals'].extend(target['terminals']   )
    src['timeouts'].extend(target['timeouts']    )
    src['infos/goal'].extend(target['infos/goal']  )
    src['infos/qpos'].extend(target['infos/qpos']  )
    src['infos/qvel'].extend(target['infos/qvel']  )



def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def load_policy(policy_file):
    data = torch.load(policy_file, map_location=torch.device('cpu'))
    policy = data['exploration/policy'].to('cpu')
    env = data['evaluation/env']
    print("Policy loaded")
    return policy, env

def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name+ '_episode_{}'.format(episode_id))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))


def distanceGoal(state, goal) -> float:
    """Computes the distance from state to goal"""
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    return np.linalg.norm([dx, dy])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env', type=str, default='Ant', help='Environment type')
    parser.add_argument('--policy_file', type=str, default='policy_file', help='file_name')
    parser.add_argument('--num_trajectories', type=int, default=int(1e2), help='Number of trajectories to collect')
    parser.add_argument('--render', "-r", action='store_true', help='record video')
    parser.add_argument("--timelimit", action='store_true', help="Turn each trajectory into fix horizon")


    args = parser.parse_args()

    maze = maze_env.U_MAZE

    config = dict(
        multi_start = False,
        v2_resets = False,
        max_episode_steps = 300,
        n_trajs = args.num_trajectories,
    )
    ic(config)
    
    if args.env == 'Ant':
        env = NormalizedBoxEnv(ant.AntMazeEnv(
            maze_map=maze, maze_size_scaling=4.0, non_zero_reset=False, v2_resets=False))
    else:
        raise NotImplementedError
    
    # env.set_target()
    env = AntMazeFixStartAndGoalWrapper(env)
    ic(env.unwrapped._goal)
    
    
    s = env.reset()
    assert (s[0], s[1]) == (0, 0), "start position is not (0, 0)"
    
    act = env.action_space.sample()
    done = False

    # Load the policy
    policy, train_env = load_policy(args.policy_file)

    # Define goal reaching policy fn
    def _goal_reaching_policy_fn(obs, goal):
        goal_x, goal_y = goal
        obs_new = obs[2:-2]
        goal_tuple = np.array([goal_x, goal_y])

        # normalize the norm of the relative goals to in-distribution values
        goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

        new_obs = np.concatenate([obs_new, goal_tuple], -1)
        return policy.get_action(new_obs)[0], (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])      

    data = reset_data()
    data["count"] = 0

    # create waypoint generating policy integrated with high level controller
    data_collection_policy = env.create_navigation_policy(
        _goal_reaching_policy_fn,
    )

    
    while data["count"] < args.num_trajectories:
        s = env.reset()
        ic(env.target_goal)
        assert (s[0], s[1]) == (0, 0), "start position is not (0, 0)"
        assert env.target_goal == (0,8)

        done = False
        timeout = False
        ts = 0
        is_success = False
        
        episode_data = reset_data()
        
        while not done:
            ic(data["count"], ts)
            
            act, waypoint_goal = data_collection_policy(s)

            if args.noisy:
                act = act + np.random.randn(*act.shape)*0.2
                act = np.clip(act, -1.0, 1.0)

            assert (act<-1).any() == False, "action is not in [-1, 1]"
            assert (act>1).any() == False, "action is not in [-1, 1]"
        
            ns, r, done, info = env.step(act)
            if args.render:
                env.render()    
            
            is_success = distanceGoal(s[:2], env.target_goal) < 0.5  # goal threshold
            
            if not args.timelimit:
                done = is_success
            
            if ts >= config["max_episode_steps"]:
                timeout = True
                done = True
                

            append_data(episode_data, s[:-2], act, r, env.target_goal, done, timeout, env.physics.data)
            
            ts += 1
            s = ns
        
        if is_success:
            extend_data(src=data, target=episode_data)
            data["count"] += 1
    
    
    print("number of trajectories collected: {}".format(data["count"]))
    assert data["count"] == args.num_trajectories, "number of trajectories collected is not equal to num_trajectories"
    data.pop("count")
    
    tl = "_timelimit" if args.timelimit else ""
    if args.noisy:
        fname = args.env + f"_maze_umaze_noisy_fixstart_fixgoal{tl}.hdf5"
    else:
        fname = args.env + f"maze_umaze_fixstart_fixgoal_{tl}.hdf5" 
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

if __name__ == '__main__':
    main()