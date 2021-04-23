import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

import pytorch_utils as ptu

class LinearCost:
    def __init__(self):
        # Initialize true weight and learned weight for cost function
        self.true_weight = np.array([0.1, 0.9])
        self.weight = np.array([0.9, 0.1])

        self.attractor = np.array([0., 1.])
        self.goal = np.array([1., 1.])

    def feature(self, state):
        """
        Returns the feature vector of a state (x, y)
        Feature 1 is the distance to goal (1, 1)
        Feature 2 is the distance to an attractor at (1, 0)
        """
        f1 = np.linalg.norm(state - self.goal)
        f2 = np.linalg.norm(state - self.attractor)
        return np.array([f1, f2])

    def true_cost(self, state):
        """ True cost to generate expert trajectories """
        return self.true_weight.T @ self.feature(state)

    def calc_cost(self, state):
        """ Agent cost function for arriving at state """
        return self.weight.T @ self.feature(state)

    def calc_feature_counts(self, trajs):
        """
        Calculate the average feature vector
        """
        feature = [self.feature(state) for traj in trajs for state in traj]
        return np.mean(feature, axis=0)

    def update(self, agent_trajs, expert_trajs, lr, normalize=False):
        """
        Update the weights with exponentiated stochastic gradient ascent
        """

        # Calculate feature counts for agent and expert trajectories
        agent_feature = self.calc_feature_counts(agent_trajs) 
        expert_feature = self.calc_feature_counts(expert_trajs)

        agent_feature += np.ones_like(agent_feature) * 0.1
        # gradient is the feature difference
        grad = agent_feature - expert_feature 

        self.weight = self.weight * np.exp(lr * grad)
        if normalize:        
            self.weight /= self.weight.sum() 

        print("Feature difference: {:04f}".format(np.linalg.norm(agent_feature - expert_feature)))
        print("Learned weight:", self.weight)

    def update_max_margin(self, agent_trajs, expert_trajs, lr, normalize=True):
        pass

class MLPCost(nn.Module):
    def __init__(self, state_dim, n_layers, size, lr):
        super(MLPCost, self).__init__()
        
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.size = size
        self.lr = lr

        self.goal = np.array([1., 1.])

        # MLP cost function
        self.cost_fn = ptu.build_mlp(
            input_size=self.state_dim,
            output_size=1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu',
            output_activation='relu' 
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.cost_fn.parameters(),
            self.lr
        )

        # true cost
        self.true_weight = np.array([0.1, 0.9])

    def feature(self, state):
        """
        Returns the feature vector of a state (x, y)
        Feature 1 is the distance to goal (0, 0)
        Feature 2 is the distance to an attractor at (-1, 0)
        """
        f1 = np.linalg.norm(state - self.goal)
        f2 = np.linalg.norm(state - np.array([0, 1]))
        return np.array([f1, f2])

    def true_cost(self, state):
        """ True cost to generate expert trajectories """
        return np.array([0.1, 0.9]).T @ self.feature(state)

    def calc_cost(self, state):
        """ Agent cost function for arriving at state """
        cost = ptu.to_numpy(self(np.array(state)))
        return cost

    def forward(self, x):
        """
        Computes the cost at state x using MLP
        """
        x = ptu.from_numpy(x)
        c = self.cost_fn(x)
        c += 1e-6
        return c

    def update(self, agent_trajs, expert_trajs):
        """
        Updates the cost parameters using agent and expert trajectories
        """ 

        agent_costs = self.calc_traj_cost(agent_trajs) 
        expert_costs = self.calc_traj_cost(expert_trajs)

#        print("Agent cost:", ptu.to_numpy(agent_costs))
#        print("Expert cost:", ptu.to_numpy(expert_costs))

        # add feature difference as margin to loss
#        agent_features = ptu.from_numpy(self.calc_feature_counts(agent_trajs))
#        expert_features = ptu.from_numpy(self.calc_feature_counts(expert_trajs))
#        margin = torch.linalg.norm(agent_features - expert_features)
#        print("Margin:", ptu.to_numpy(margin))
        
        loss = torch.mean(expert_costs) + torch.log(torch.mean(torch.exp(-agent_costs)))
        #loss = torch.mean(expert_costs) - torch.mean(agent_costs) - margin

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_log = {"Training_loss": ptu.to_numpy(loss)}
        print("Training_loss:", ptu.to_numpy(loss))
        return train_log

    def calc_feature_counts(self, trajs):
        """
        Calculate the average feature vector
        """
        feature = [self.feature(state) for traj in trajs for state in traj]
        return np.mean(feature, axis=0)

    def calc_traj_cost(self, trajs):
        """
        Calculate the cost of a list of trajectories
        The return costs are flattened
        """
        trajs = np.array([state for traj in trajs for state in traj])
        costs = self.forward(trajs)
        print(trajs)
        print(costs)
        return costs

#class CostNet(nn.Module):
#    def __init__(self):
#        super().__init__()#

#        self.dtype = torch.float32
#        self.device = 'cpu'
#        self.weight = nn.Parameter(torch.tensor([1., 0.], dtype=self.dtype, device=self.device))
#        self.goal = torch.tensor([0., 0.], dtype=self.dtype, device=self.device)#

#    def forward(self, state):
#        return F.softmax(self.weight).T @ self.feature(state)#
#

#    def feature(self, state):
#        """
#        Returns the feature vector of a state (x, y)
#        Feature 1 is the distance to goal (0, 0)
#        Feature 2 is the Mahalanobis distance to the Gaussian at (-0.5, -0.5)
#        """
#        f1 = torch.linalg.norm(state - self.goal)
#        #f2 = np.sqrt((state-self.mu) @ self.inv_cov @ (state-self.mu).T)
#        f2 = np.linalg.norm(state - torch.tensor([-1., 0.], dtype=self.dtype, device=self.device))
#        return torch.tensor([f1, f2], dtype=self.dtype, device=self.device)#
#

#    def update(self, agent_trajs, expert_trajs):
#        """
#        Update the cost weights given agent and expert trajectories
#        Objective is to maximize the expert trajectory likelihood
#        """#

#        # Loss is the negative log likelihood of 



