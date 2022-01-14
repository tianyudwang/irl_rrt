import itertools
import numpy as np
import torch
from torch import nn
from torch import optim

import irl.utils.pytorch_util as ptu

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
            # The input is current state and next state -> input size = ob_dim + ob_dim
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

    def forward(self, x1, x2):
        """
        Computes the reward of motion from x1 to x2 using MLP
        x1, x2 are np.array (batch_size, ob_dim) and r is torch.tensor (batch_size, 1)
        This implementation ensures learned cost (-r(x1, x2)) is a metric
        """
        x = torch.cat((ptu.from_numpy(x1), ptu.from_numpy(x2)), dim=-1)
        # * output of mlp is cost
        r = -self.mlp(x)
#        h1 = self.mlp(ptu.from_numpy(x1))
#        h2 = self.mlp(ptu.from_numpy(x2))
#        r = -torch.linalg.norm(h1 - h2, dim=-1, keepdim=True)    # ensure reward is non-positive
        return r

    def cost_fn(self, x1, x2):
        """
        Compute cost for motion between state x1 and next state x2
        """
        return -self.reward_fn(x1, x2)

    def reward_fn(self, x1, x2):
        """
        Compute reward for motion between state x1 and next state x2
        """
        x1 = x1.reshape(1, self.ob_dim)
        x2 = x2.reshape(1, self.ob_dim)
        r = ptu.to_numpy(self(x1, x2)).item()
        return r

    def update(self, demo_paths, agent_paths, agent_log_probs):
        """
        Reward function update
        """
        demo_rewards = self.calc_path_rewards(demo_paths)
        agent_rewards = self.calc_path_rewards(agent_paths)
        agent_log_probs = torch.unsqueeze(ptu.from_numpy(agent_log_probs), dim=-1)

        demo_Q = torch.sum(demo_rewards, dim=2, keepdim=False)
        agent_Q = torch.sum(agent_rewards, dim=2, keepdim=False)

        demo_loss = -torch.sum(demo_Q)
        agent_loss = torch.sum(agent_Q)
        #agent_loss = torch.sum(torch.log(torch.mean(torch.exp(agent_Q - agent_log_probs), dim=1)))

        loss = demo_loss + agent_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_reward_log = {
            "Reward_loss": ptu.to_numpy(loss)
        }

        print(f"Reward training loss: {loss.item():.2f}, {demo_loss.item():.2f}, {agent_loss.item()}") 
        return train_reward_log
        
    def calc_path_rewards(self, paths):
        """
        Calculate the rewards of transitions
        """
        states, next_states = paths[:,:,:-1,:], paths[:,:,1:,:]
        rewards = self.forward(states, next_states)
        
        return rewards