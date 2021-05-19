import itertools
import numpy as np
import torch
from torch import nn
from torch import optim

from irl.scripts import pytorch_util as ptu

class MLPReward(nn.Module):
    """
    Defines a reward function given the current observation and action
    """
    def __init__(self, ob_dim, ac_dim, n_layers, size, output_size, learning_rate):
        super().__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_layers = n_layers
        self.size = size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.mlp = ptu.build_mlp(
            input_size=self.ob_dim * 2,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu',
            output_activation='sigmoid'
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.mlp.parameters(),
            self.learning_rate
        )

    def forward(self, x, nx):
        """
        Computes the reward of motion from x to nx using MLP
        x, nx are np.array and r is torch.tensor
        """
        x = ptu.from_numpy(x)
        nx = ptu.from_numpy(nx)
        x = torch.cat((x, nx), dim=-1)
        # Ensure the reward is non-positive with sigmoid activation in last layer
        r = -self.mlp(x)
        return r

    def cost_fn(self, state, next_state):
        """
        Compute cost for motion between state and next state
        """
        state = state.reshape(1, self.ob_dim)
        next_state = next_state.reshape(1, self.ob_dim)
        cost = -ptu.to_numpy(self(state, next_state)).item()
        return cost

    def update(self, demo_paths, agent_paths):
        demo_costs = self.calc_path_cost(demo_paths)
        agent_costs = self.calc_path_cost(agent_paths)

        loss = torch.mean(demo_costs) - torch.mean(agent_costs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_reward_log = {
            "Reward_loss": ptu.to_numpy(loss)
        }
        print("Reward training loss:", ptu.to_numpy(loss))
        return train_reward_log
        
    def calc_path_cost(self, paths):
        """
        Calculate the cost of a list of paths
        The returned costs are flattened
        """
        states = np.array([path[i] for path in paths for i in range(len(path)-1)])
        next_states = np.array([path[i+1] for path in paths for i in range(len(path)-1)])
        costs = self.forward(states, next_states)
        return costs

#    def update(self, demo_obs, demo_acs, sample_obs, sample_acs, log_probs):
#        """
#        Computes the loss and updates the reward parameters
#        Objective is to maximize sum of demo rewards and minimize sum of sample rewards
#        Use importance sampling for policy samples
#        Recall that the MLE objective to maximize is:
#            1/N sum_{i=1}^N return(tau_i) - log Z
#          = 1/N sum_{i=1}^N return(tau_i) - log E_{tau ~ p(tau)} [exp(return(tau))]
#          = 1/N sum_{i=1}^N return(tau_i) - log E_{tau ~ q(tau)} [p(tau) / q(tau) * exp(return(tau))]
#          = 1/N sum_{i=1}^N return(tau_i) - log (sum_j exp(return(tau_j)) * w(tau_j) / sum_j w(tau_j))
#        where w(tau) = p(tau) / q(tau) = 1 / prod_t pi(a_t|s_t) 
#        """
#        demo_obs = ptu.from_numpy(demo_obs)
#        demo_acs = ptu.from_numpy(demo_acs)
#        sample_obs = ptu.from_numpy(sample_obs)
#        sample_acs = ptu.from_numpy(sample_acs)
#        #log_probs = torch.squeeze(ptu.from_numpy(log_probs), dim=-1)#

#        demo_costs = -self(demo_obs, demo_acs)
#        sample_costs = -self(sample_obs, sample_acs)
#        # using 1/N sum_{i=1}^N return(tau_i) - log 1/M (sum_j exp(return(tau_j)) / prod_t pi(a_t|s_t) )
##        w = sample_return - torch.sum(log_probs, dim=1)
##        w_max = torch.max(w)#

#        # TODO: Use importance sampling to estimate sample return 
#        # trick to avoid overflow: log(exp(x1) + exp(x2)) = x + log(exp(x1-x) + exp(x2-x)) where x = max(x1, x2)
#        #loss = -torch.mean(demo_return) + torch.log(torch.sum(torch.exp(w-w_max))) + w_max
#        
#        print(torch.mean(demo_costs).item(), torch.mean(sample_costs).item())
#        #loss = torch.mean(demo_costs) - torch.mean(sample_costs) #

#        probs = 1
#        loss = torch.mean(demo_costs) + torch.log(torch.mean(torch.exp(-sample_costs)/(probs+1e-7)))
#        
#        self.optimizer.zero_grad()
#        loss.backward()
#        self.optimizer.step()#

#        train_reward_log = {
#            "Reward_loss": ptu.to_numpy(loss)
#        }
#        print("Reward training loss:", ptu.to_numpy(loss))
#        return train_reward_log