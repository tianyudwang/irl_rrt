import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse

from irl.utils.wrappers import Maze2DFixedStartWrapper

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_trajectories', type=int, default=int(1e2), help='Number of trajectories to collect')
    args = parser.parse_args()


    env = gym.make(args.env_name)
    env = Maze2DFixedStartWrapper(env)
    maze = env.str_maze_spec
    max_episode_steps = env.env._max_episode_steps
    target = env.unwrapped._target

    state_low = np.array([0.5, 0.5, -5., -5.])
    state_high = np.array([3.5, 3.5, 5., 5.])

    data = reset_data()

    for _ in range(args.num_trajectories):
        s = env.reset()
        s = np.clip(s, state_low, state_high)
        done = False
        ts = 0
        controller = waypoint_controller.WaypointController(maze)

        while not done:
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, target)
            if args.noisy:
                act = act + np.random.randn(*act.shape)*0.5

            act = np.clip(act, -1.0, 1.0)
            
            if ts >= max_episode_steps:
                done = True
            append_data(data, s, act, target, done, env.sim.data)

            s, _, _, _ = env.step(act)
            s = np.clip(s, state_low, state_high)

            ts += 1 

            if args.render:
                env.render()
    
    if args.noisy:
        fname = '%s-noisy.hdf5' % args.env_name
    else:
        fname = '%s.hdf5' % args.env_name
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
