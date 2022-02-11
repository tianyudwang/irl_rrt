import itertools
from typing import Dict, Union, Optional, List

import numpy as np
import torch as th
from torch import nn
from torch import optim

import irl.utils.pytorch_util as ptu

class RewardNet(nn.Module):
    """
    Defines a reward function given the current observation and action
    """
    def __init__(self, reward_params: Dict[str, Union[float, int]]):
        super().__init__()

        self.reward_params = reward_params

        self.device = th.device("cuda")
        self.device_cpu = th.device("cpu")

        self.model = self.init_model(device=self.device)
        self.model_cpu = self.init_model(device=self.device_cpu)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.reward_params['learning_rate'],
            weight_decay=0.1,
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
        """
        with th.no_grad():
            x1, x2 = self.preprocess_input(x1, x2, self.device_cpu)
            reward = self(self.model_cpu, x1, x2)
        return reward.item()

    def lcr_regularizer(self, path):
        states, next_states = path[:-1], path[1:]
        rewards = self(self.model, states, next_states)
        lcr = th.sum(th.square(rewards[1:] - rewards[:-1]), dim=0, keepdim=False)
        return lcr


    # def update(
    #         self,
    #         demo_paths: List[List[np.ndarray]],
    #         agent_paths: List[List[np.ndarray]],
    #         agent_log_probs: List[np.ndarray],
    #         local_constant_rate: Optional[bool] = True
    #     ) -> Dict[str, float]:
    #     """Optimize the reward function
    #     The loss function for reward parameters is
    #     J_r(theta) = E_{(s,a)~D}[Q*(s, a) - log(E_{a'~pi(s)}[e^(Q*(s, a')) / pi(a'|s)])],
    #     which is hard to optimize due to numerical instability in 1 / pi(a'|s)
    #     Instead, we are using
    #     J_r(theta) = E_{(s,a)~D}[Q*(s, a) - E_{a'~pi(s)}[Q*(s, a')]] 
    #     """

    #     Q_diff = []
    #     for demo_path, agent_path, agent_log_prob in zip(
    #         demo_paths, agent_paths, agent_log_probs
    #     ):
    #         demo_Q = th.cat([self.compute_Q(path) for path in demo_path], dim=0)
    #         agent_Q = th.cat([self.compute_Q(path) for path in agent_path], dim=0)

    #         assert len(agent_log_prob) == len(agent_path)
    #         agent_log_prob = agent_log_prob.astype(np.float32)
    #         assert agent_log_prob.dtype == np.float32, "agent_log_prob dtype is not np.float32"
    #         agent_log_prob = th.from_numpy(agent_log_prob).to(self.device)

    #         demo_Q = th.mean(demo_Q, dim=0, keepdim=True)
    #         # agent_Q = th.mean(agent_Q, dim=0, keepdim=True)

    #         # log(E_{a'~pi(s)}[e^(Q*(s, a')) / pi(a'|s)])] = 
    #         # log(E_{a'~pi(s)}[e^(Q*(s, a') - log pi(a'|s))])]
    #         agent_Q = th.log(th.mean((th.exp(agent_Q - agent_log_prob)), dim=0, keepdim=True))
    #         Q_diff.append(demo_Q - agent_Q)
    #         print(f"demo_Q {demo_Q.item():.2f}, agent_Q {agent_Q.item():.2f}, agent_log_prob {agent_log_prob.item():.2f}")
    #     reward_loss = th.mean(th.cat(Q_diff, dim=0))

    #     demo_lcr_loss, agent_lcr_loss = [], []
    #     if local_constant_rate:
    #         for demo_path, agent_path in zip(demo_paths, agent_paths):
    #             demo_lcr_loss.append(
    #                 th.cat([self.lcr_regularizer(path) for path in demo_path], dim=0)
    #             )
    #             agent_lcr_loss.append(
    #                 th.cat([self.lcr_regularizer(path) for path in agent_path], dim=0)
    #             )
    #     demo_lcr_loss = th.mean(th.cat(demo_lcr_loss, dim=0))
    #     agent_lcr_loss = th.mean(th.cat(agent_lcr_loss, dim=0))

    #     loss = reward_loss + demo_lcr_loss + agent_lcr_loss


    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     train_reward_log = {
    #         "Reward_loss": ptu.to_numpy(reward_loss), 
    #         "demo_lcr_loss": ptu.to_numpy(demo_lcr_loss),
    #         "agent_lcr_loss": ptu.to_numpy(agent_lcr_loss)
    #     }

    #     output_str = ""
    #     for loss_name, loss_val in train_reward_log.items():
    #         output_str += loss_name + f" {loss_val.item():.2f} " 
    #     print("\n", output_str)
    #     return train_reward_log


    def update(
            self,
            demo_paths: List[th.Tensor],
            agent_paths_l: List[List[th.Tensor]],
            agent_log_probs_l: List[th.Tensor]
        ) -> Dict[str, float]:
        """Optimize reward neural network"""

        demo_Qs, agent_Qs, agent_lse_Qs = [], [], []
        for i in range(len(demo_paths)):
            demo_Q = self.compute_Q(demo_paths[i])
            agent_Q = th.cat([self.compute_Q(paths[i]) for paths in agent_paths_l])
            agent_log_prob = th.stack([log_probs[i] for log_probs in agent_log_probs_l])

            assert agent_Q.shape == agent_log_prob.shape
            # agent_Q = th.mean(agent_Q, dim=0, keepdim=True)
            agent_lse_Q = th.log(th.mean((th.exp(agent_Q - agent_log_prob)), dim=0, keepdim=True))

            demo_Qs.append(demo_Q)
            agent_Qs.append(agent_Q)
            agent_lse_Qs.append(agent_lse_Q)
        
        # Reward loss
        demo_Qs = th.mean(th.cat(demo_Qs, dim=0))
        agent_Qs = th.mean(th.cat(agent_lse_Qs, dim=0))
        reward_loss = - demo_Qs + agent_Qs

        # Regularization loss
        demo_lcr_loss = [self.lcr_regularizer(path) for path in demo_paths]
        agent_lcr_loss = [
            self.lcr_regularizer(path) for agent_paths in agent_paths_l 
            for path in agent_paths 
        ]
        demo_lcr_loss = th.sum(th.cat(demo_lcr_loss, dim=0))
        agent_lcr_loss = th.sum(th.cat(agent_lcr_loss, dim=0))

        loss = reward_loss + demo_lcr_loss + agent_lcr_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_reward_log = {
            "Reward/demo_Q": np.mean(ptu.to_numpy(demo_Qs)),
            "Reward/agent_Q": np.mean(ptu.to_numpy(agent_Qs)),
            "Reward/agent_log_prob": np.mean(ptu.to_numpy(th.cat(agent_log_probs_l))),
            "Reward/reward_loss": ptu.to_numpy(reward_loss), 
            "Reward/demo_lcr_loss": ptu.to_numpy(demo_lcr_loss),
            "Reward/agent_lcr_loss": ptu.to_numpy(agent_lcr_loss),
        }

        for loss_name, loss_val in train_reward_log.items():
            print(loss_name, f" {loss_val.item():.2f}")
        return train_reward_log

    # def compute_Q(
    #         self, 
    #         path: np.ndarray, 
    #         device: Optional[th.device] = th.device('cuda')
    #     ) -> th.Tensor:
    #     """Compute the Q value of a planned path"""
    #     assert (len(path.shape) == 2 and path.shape[1] == self.reward_params['ob_dim']), \
    #         "path variable does not have shape (N, state_dim)"

    #     if device == self.device:
    #         states, next_states = self.preprocess_input(path[:-1], path[1:], self.device)
    #         reward = self(self.model, states, next_states)
    #     else:
    #         states, next_states = self.preprocess_input(path[:-1], path[1:], self.device_cpu)
    #         reward = self(self.model_cpu, states, next_states)

    #     Q = th.sum(reward, dim=0, keepdim=False)
    #     return Q

    def compute_Q(
            self, 
            path: th.Tensor
        ) -> th.Tensor:
        """Compute the Q value of a path"""
        states, next_states = path[:-1], path[1:]
        reward = self(self.model, states, next_states)
        Q = th.sum(reward, dim=0, keepdim=False)
        return Q



    ###############################################
    def copy_model_to_cpu(self) -> None:
        PATH = '/tmp/irl_reward_net.pt'
        th.save(self.model.state_dict(), PATH)
        self.model_cpu.load_state_dict(th.load(PATH, map_location=self.device_cpu))

    def save(self, filename: str) -> None:
        th.save(self.model.state_dict(), filename)