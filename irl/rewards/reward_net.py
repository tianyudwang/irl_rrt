import itertools
from typing import Mapping, Union, Optional, List

import numpy as np
import torch as th
from torch import nn
from torch import optim

import irl.util.pytorch_util as ptu

class RewardNet(nn.Module):
    """
    Defines a reward function given the current observation and action
    """
    def __init__(self, reward_params: Mapping[str, Union[float, int]]):
        super().__init__()

        self.reward_params = reward_params

        self.device = th.device("cuda")
        self.device_cpu = th.device("cpu")

        self.model = self.init_model(device=self.device)
        self.model_cpu = self.init_model(device=self.device_cpu)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            self.reward_params['learning_rate']
        )

    def init_model(self, device: Optional[th.device] = th.device("cuda")) -> nn.Module:
        """Initialize reward neural network"""
        model = ptu.build_mlp(
            input_size=self.reward_params['ob_dim'] * 2,
            output_size=self.reward_params['output_size'],
            n_layers=self.reward_params['n_layers'],
            size=self.reward_params['size'],
            activation='relu',
            output_activation='sigmoid'
        ).to(device)
        return model 

    def forward(
            self, 
            model: nn.Module, 
            x1: np.ndarray, 
            x2: np.ndarray
        ) -> th.FloatTensor:
        """Computes the reward of motion from x1 to x2 with an MLP"""
        x = th.cat((x1, x2), dim=-1)
        r = -model(x)
        return r

    def preprocess_input(
            self,
            x1: np.ndarray,
            x2: np.ndarray,
            device: Optional[th.device] = th.device('cuda')
        ) -> Union[th.Tensor]:
        """Load numpy array to torch tensors on cpu or cuda"""
        if len(x1.shape) < 2 or len(x2.shape) < 2:
            x1 = x1.reshape(1, self.reward_params['ob_dim'])
            x2 = x2.reshape(1, self.reward_params['ob_dim'])
        x1, x2 = x1.astype(np.float32), x2.astype(np.float32)
        assert x1.dtype == np.float32, "State x1 dtype is not np.float32"
        assert x2.dtype == np.float32, "State x2 dtype is not np.float32"

        x1 = th.from_numpy(x1).to(device)
        x2 = th.from_numpy(x2).to(device)
        return x1, x2


    def cost_fn(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute cost for motion between state x1 and next state x2
        This function is used as inference in planning and does not require autograd
        """
        return -self.reward_fn(x1, x2)

    def reward_fn(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute reward for motion between state x1 and next state x2
        This function is used as inference when collecting rollouts for SAC 
        and does not require autograd
        """
        with th.no_grad():
            x1, x2 = self.preprocess_input(x1, x2, self.device_cpu)
            reward = self(self.model_cpu, x1, x2)
        return reward.item()

    # def update(self, demo_paths, agent_paths, agent_log_probs):
    #     """
    #     Reward function update
    #     """
    #     demo_rewards = self.calc_path_rewards(demo_paths)
    #     agent_rewards = self.calc_path_rewards(agent_paths)
    #     agent_log_probs = th.unsqueeze(ptu.from_numpy(agent_log_probs), dim=-1)

    #     demo_Q = th.sum(demo_rewards, dim=2, keepdim=False)
    #     agent_Q = th.sum(agent_rewards, dim=2, keepdim=False)

    #     demo_loss = -th.sum(demo_Q)
    #     agent_loss = th.sum(agent_Q)
    #     #agent_loss = th.sum(torch.log(torch.mean(torch.exp(agent_Q - agent_log_probs), dim=1)))

    #     loss = demo_loss + agent_loss
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     train_reward_log = {"Reward_loss": ptu.to_numpy(loss)}

    #     print("Reward training loss:", loss.item(), demo_loss.item(), agent_loss.item())
    #     return train_reward_log

    # def calc_path_rewards(self, paths: np.ndarray) -> th.Tensor:
    #     """Calculate the rewards of transitions, requires autograd"""
    #     states, next_states = self.preprocess_input(
    #         paths[:,:,:-1,:], 
    #         paths[:,:,1:,:], 
    #         self.device
    #     )
    #     rewards = self(self.model, states, next_states)
    #     return rewards
    
    def update(
            self,
            demo_paths: List[List[np.ndarray]],
            agent_paths: List[List[np.ndarray]],
            agent_log_probs: List[np.ndarray]
        ) -> Mapping[str, float]:
        """Optimize the reward function
        The loss function for reward parameters is
        J_r(theta) = E_{(s,a)~D}[Q*(s, a) - log(E_{a'~pi(s)}[e^(Q*(s, a')) / pi(a'|s)])],
        which is hard to optimize due to numerical instability in 1 / pi(a'|s)
        Instead, we are using
        J_r(theta) = E_{(s,a)~D}[Q*(s, a) - E_{a'~pi(s)}[Q*(s, a')]] 
        """

        Q_diff = []
        for demo_path, agent_path in zip(demo_paths, agent_paths):
            demo_Q = th.cat([self.compute_Q(path) for path in demo_path], dim=0)
            agent_Q = th.cat([self.compute_Q(path) for path in agent_path], dim=0)
            demo_Q = th.mean(demo_Q, dim=0, keepdim=True)
            agent_Q = th.mean(agent_Q, dim=0, keepdim=True)
            Q_diff.append(demo_Q - agent_Q)
        loss = th.mean(th.cat(Q_diff, dim=0))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_reward_log = {"Reward_loss": ptu.to_numpy(loss)}

        print("Reward training loss:", loss.item())
        return train_reward_log


    def compute_Q(
            self, 
            path: np.ndarray, 
            device: Optional[th.device] = th.device('cuda')
        ) -> th.Tensor:
        """Compute the Q value of a planned path"""
        assert (len(path.shape) == 2 and path.shape[1] == self.reward_params['ob_dim']), \
            "path variable does not have shape (N, state_dim)"

        if device == self.device:
            states, next_states = self.preprocess_input(path[:-1], path[1:], self.device)
            reward = self(self.model, states, next_states)
        else:
            states, next_states = self.preprocess_input(path[:-1], path[1:], self.device_cpu)
            reward = self(self.model_cpu, states, next_states)

        Q = th.sum(reward, dim=0, keepdim=False)
        return Q



    ###############################################
    def copy_model_to_cpu(self) -> None:
        PATH = '/tmp/irl_reward_net.pt'
        th.save(self.model.state_dict(), PATH)
        self.model_cpu.load_state_dict(th.load(PATH, map_location=self.device_cpu))