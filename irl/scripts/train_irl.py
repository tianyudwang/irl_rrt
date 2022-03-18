from typing import List

import argparse
import os
import os.path as osp
import time

from irl.agents.irl_agent import IRL_Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Reacher-v2')
    parser.add_argument('--expert_policy', type=str, default='SAC_Reacher-v2')    
    parser.add_argument('--suffix', type=str, default=None)

    parser.add_argument(
        '--n_iter', '-n', type=int, default=200,
        help='Number of total iterations')
    parser.add_argument(
        '--demo_size', type=int, default=100, 
        help='Number of expert demonstration trajectories to collect'
    )
    parser.add_argument(
        '--reward_update_batch_size', type=int, default=32, 
        help='Number of transitions for reward training'
    )
    parser.add_argument('--agent_action_from_demo_state', '-aa', action='store_true')
    parser.add_argument(
        '--policy_update_batch_size', type=int, default=32, 
        help='Number of episodes for policy training'
    )
    parser.add_argument(
        '--n_reward_updates_per_itr', type=int, default=2, 
        help='Number of reward updates per iteration'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=32,
        help='Number of policy rollout episodes to collect for evaluation'
    )

    # NN parameters
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--activation', '-a', type=str, default='relu')
    parser.add_argument('--output_size', '-os', type=int, default=1)
    parser.add_argument('--output_activation', '-oa', type=str, default='relu')
    parser.add_argument('--lcr_reg', type=float, default=0)
    parser.add_argument('--gail_reg', type=float, default=0)
    parser.add_argument('--grad_norm', type=float, default=-1)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true')
    
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(os.path.realpath(__file__)))), 'data')


    logdir = args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = osp.join(data_path, logdir)
    if params['suffix']:
        logdir += '_' + params['suffix']
    params['logdir'] = logdir
    print(params)

    # dump params
    os.makedirs(logdir, exist_ok=True)
    import json
    with open(osp.join(logdir, 'params.json'), 'w') as fp:
        json.dump(params, fp)

    ###################
    ### RUN TRAINING
    ###################

    irl_model = IRL_Agent(params)
    irl_model.train()

if __name__ == '__main__':
    main()