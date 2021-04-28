import os
import argparse
import numpy as np
from multiprocessing import Pool

import torch

from rrt_star import RRTStar
from cost_models import MLPCost
import pytorch_utils as ptu
import utils

class RRTIRL_Trainer:
    """
    Maximum entropy inverse reinforcement learning with RRT* planner
    """

    def __init__(self, params):

        self.params = params

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # expert and agent cost 
        self.cost = MLPCost(2, 2, 64, self.params['lr'])

        # training
        self.lr = utils.LinearDecayLR(params['lr'])
        self.logger = utils.Logger(self.cost.true_weight)


    def training_loop(self):

        # Collect expert demo
        expert_trajs = self.collect_trajs(self.params['num_expert_trajs'], self.cost.true_cost)
        if self.params['visualize_trajs']:
            filename = '../imgs/nn_expert_trajs.png'
            utils.visualize_trajs(expert_trajs, filename)

        # main training loop
        for i in range(self.params['num_iters']):
            print("#"*20)
            print("Iteration: {}/{}".format(i+1, self.params['num_iters']))

            # Collect agent trajectories under current cost
            agent_trajs = self.collect_trajs(self.params['num_agent_trajs_per_iter'], self.cost.calc_cost)

            # Calculate gradient and update weights
            for _ in range(self.params['num_reward_train_steps_per_iter']):
                self.cost.update(agent_trajs, expert_trajs)

            # Logging
            #self.logger.log(i, self.cost.weight, np.linalg.norm(agent_feature - expert_feature))
            if self.params['visualize_trajs'] and i % 1 == 0:
                filename = '../imgs/nn_agent_trajs_iter{}.png'.format(i)
                utils.visualize_trajs(agent_trajs, filename)

#    def collect_traj(self, cost_fn):
#        rrt_star = RRTStar(
#            start=[0, 0],
#            goal=[1, 1],
#            obstacle_list=[],
#            rand_area=[0, 1],
#            expand_dis=self.params['expand_dis'],
#            path_resolution=self.params['path_resolution'],
#            cost_fn=cost_fn,
#            connect_circle_dist=self.params['connect_circle_dist'],
#            goal_sample_rate=self.params['goal_sample_rate'],
#            max_iter=10000)
#        path = rrt_star.planning(animation=False)
#        return [np.array(state) for state in path]#
#

#    def collect_trajs(self, num_trajs, cost_fn):
#        """
#        Collect trajectories according to the cost function using RRT*
#        Args:
#            num_trajs: Number of trajectories to collect
#            cost_fn: Function that returns a cost for each state
#        """#

#        # Use a pool of worker to run RRT* in parallel
#        # Cannot use with GPU
#        with Pool(4) as pool:
#            paths = pool.map_async(self.collect_traj, [cost_fn]*num_trajs)
#            paths = paths.get()
#        return paths

    def collect_trajs(self, num_trajs, cost_fn):
        """
        Collect trajectories according to the cost function using RRT*
        Args:
            num_trajs: Number of trajectories to collect
            cost_fn: Function that returns a cost for each state
        """
        paths = []
        while len(paths) < num_trajs:
            print('Collecting trajectory: {}/{}'.format(len(paths), num_trajs))
            rrt_star = RRTStar(
                start=[0, 0],
                goal=[1, 1],
                obstacle_list=[],
                rand_area=[0, 1],
                expand_dis=self.params['expand_dis'],
                path_resolution=self.params['path_resolution'],
                cost_fn=cost_fn,
                connect_circle_dist=self.params['connect_circle_dist'],
                goal_sample_rate=self.params['goal_sample_rate'],
                max_iter=10000)
            path = rrt_star.planning(animation=False)
            path = [np.array(state) for state in path]
            paths.append(path)
        return paths


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    parser.add_argument('--num_expert_trajs', type=int, default=10)
    parser.add_argument('--visualize_trajs', action='store_true')
    parser.add_argument('--num_iters', type=int, default=100)
    parser.add_argument('--num_agent_trajs_per_iter', type=int, default=10)
    parser.add_argument(
        '--num_reward_train_steps_per_iter', type=int, default=10,
        help='Number of reward updates per iteration'
    )

    parser.add_argument('--expand_dis', type=float, default=0.03)
    parser.add_argument('--path_resolution', type=float, default=0.01)
    parser.add_argument('--goal_sample_rate', type=int, default=5)
    parser.add_argument('--connect_circle_dist', type=float, default=2.0)

    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    trainer = RRTIRL_Trainer(params)
    trainer.training_loop()

if __name__ == '__main__':
    main()