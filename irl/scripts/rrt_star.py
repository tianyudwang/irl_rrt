import math
import random
import heapq

import matplotlib.pyplot as plt
import numpy as np

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, state):
            self.state = state      # position of this node
            self.path = []          # positions from parent node to this node, used for collision check
            self.parent = None      # parent node of this node

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter
        start:Start Position np.array
        goal:Goal Position [x,y, ...]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max] min, max are np.array
        expand_dis:Maximum distance to steer from nearest node to random node to generate new node
        path_resolultion:Discretize edge for collision check during steering
        goal_sample_rate:Probability percentage to sample goal as random node
        """
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.dim = start.shape[0]
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1]) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.state)
        d = self.calc_distance(new_node, to_node)
        dir_vec = to_node.state - from_node.state
        # return from_node if there is no distance to steer
        if np.linalg.norm(dir_vec) < 1e-6:
            return from_node
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        new_node.path = [new_node.state]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)
        for i in range(n_expand):
            new_node.state = from_node.state + (i+1) * self.path_resolution * dir_vec
            new_node.path.append(new_node.state)

        d = self.calc_distance(new_node, to_node)
        if d <= self.path_resolution:
            new_node.state = to_node.state
            new_node.path.append(new_node.state)

        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [self.end.state]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.state)
            node = node.parent
        path.append(node.state)
        return path

    def calc_dist_to_goal(self, node):
        return self.calc_distance(node, self.end)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                np.random.uniform(self.min_rand, self.max_rand)
            )
        else:  # goal point sampling
            rnd = self.Node(self.end.state)
        return rnd

    def draw_graph(self, rnd=None):
        """
        Visualize the algorithm
        Only show the first two dimensions
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.state[0], rnd.state[1], "^k")
        for node in self.node_list:
            if node.parent:
                path_x = [state[0] for state in node.path]
                path_y = [state[1] for state in node.path]
                plt.plot(path_x, path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.state[0], self.start.state[1], "xr")
        plt.plot(self.end.state[0], self.end.state[1], "xr")
        plt.axis("equal")
        plt.axis([self.min_rand[0], self.max_rand[0], self.min_rand[1], self.max_rand[1]])
        plt.grid(True)
        plt.pause(0.001)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [np.linalg.norm(node.state - rnd_node.state) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def check_collision(node, obstacleList):
        if len(obstacleList) == 0:
            return True     # safe
        else:
            raise NotImplementedError
        #if node is None:
        #    return False
        #for (ox, oy, size) in obstacleList:
        #    dx_list = [ox - x for x in node.path_x]
        #    dy_list = [oy - y for y in node.path_y]
        #    d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
        #    if min(d_list) <= size**2:
        #        return False  # collision
        #return True  # safe

    #@staticmethod
    #def calc_distance_and_angle(from_node, to_node):
    #    dx = to_node.x - from_node.x
    #    dy = to_node.y - from_node.y
    #    d = math.hypot(dx, dy)
    #    theta = math.atan2(dy, dx)
    #    return d, theta

    @staticmethod
    def calc_distance(from_node, to_node):
        return np.linalg.norm(from_node.state - to_node.state)

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, state):
            super().__init__(state)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 cost_fn=None):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        cost_fn:Cost function for arriving at state s
        """
        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal)
        self.search_until_max_iter = search_until_max_iter
        self.cost_fn = cost_fn

    def planning(self, animation=True):
        """
        rrt star path planning
        animation: flag for animation on or off .
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + self.calc_stage_cost(near_node, new_node)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd)

            if ((not self.search_until_max_iter) and new_node):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and the tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node
            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None
        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)
        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        return new_node
        
    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the tree that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        # TODO: change the search radius
        r = self.connect_circle_dist * (math.log(nnode) / nnode) ** (1. / self.dim)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [np.linalg.norm(node.state - new_node.state) for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree
                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost
            if no_collision and improved_cost:
                near_node.state = edge_node.state
                near_node.cost = edge_node.cost
                near_node.path = edge_node.path
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d = self.calc_stage_cost(from_node, to_node)
        return from_node.cost + d

    def calc_stage_cost(self, from_node, to_node):
        d = self.calc_distance(from_node, to_node)
        if self.cost_fn is not None:
            d = (self.cost_fn(from_node.state) + self.cost_fn(to_node.state)) / 2.0 * d
        return d

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

def main():
    #min_rand = np.array([-1., -1., -8.])
    #max_rand = np.array([1., 1., 8.])
    #start = np.random.uniform(low=min_rand, high=max_rand)
    #goal = np.array([0., 0., 0.])

    min_rand = np.array([-1., -1.])
    max_rand = np.array([1., 1.])
    start = np.array([-.9, -.9])
    goal = np.array([.9, .9])

#    rrt = RRT(
#        start=start,
#        goal=goal,
#        obstacle_list=[],
#        rand_area=[min_rand, max_rand],
#        expand_dis=0.1,
#        path_resolution=0.04,
#        goal_sample_rate=10,
#        max_iter=10000
#    )
#    path = rrt.planning(animation=True)

    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=[],
        rand_area=[min_rand, max_rand],
        expand_dis=0.1,
        path_resolution=0.05,
        cost_fn=None,
        connect_circle_dist=1,
        goal_sample_rate=10,
        max_iter=10000,
        search_until_max_iter=True)
    path = rrt_star.planning(animation=True)

if __name__ == '__main__':
    main()