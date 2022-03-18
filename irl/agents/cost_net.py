from typing import Optional, List, Dict, Any

import numpy as np
import torch as th
from torch import nn
from torch import optim

from stable_baselines3.common.logger import Logger 
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from irl.utils import utils, types
import irl.utils.pytorch_utils as ptu


class CostNet(nn.Module):
    """
    Defines a cost function given the current observation and action
    """
    def __init__(
        self, 
        params: Dict[str, Any],
        logger: Logger,
    ):
        super().__init__()

        self.params = params
        self.model = ptu.build_mlp(
            # input_size=ob_dim+ac_dim,
            input_size=self.params['ob_dim']*2,
            output_size=self.params['output_size'],
            n_layers=self.params['n_layers'],
            size=self.params['size'],
            activation=self.params['activation'],
            output_activation=self.params['output_activation']
        ).to('cuda')

        self.model = self.build_mlp(device='cuda')
        self.model_cpu = self.build_mlp(device='cpu')

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay'],
        )

        self.logger = logger

        self.lcr_reg_coeff = self.params['lcr_reg']
        self.gail_reg_coeff = self.params['gail_reg']
        self.grad_norm = self.params['grad_norm']

    def build_mlp(self, device='cpu'):
        model = ptu.build_mlp(
            # input_size=ob_dim+ac_dim,
            input_size=self.params['ob_dim']*2,
            output_size=self.params['output_size'],
            n_layers=self.params['n_layers'],
            size=self.params['size'],
            activation=self.params['activation'],
            output_activation=self.params['output_activation']
        ).to(device)
        return model

    def forward(self, model: nn.Module, s: th.Tensor, ns: th.Tensor):
        s_ns = th.cat([s, ns], dim=-1)
        cost = model(s_ns)        
        return cost

    def reward(self, s: np.ndarray, ns: np.ndarray):
        """Query single step reward in gym env, do not track gradient"""
        assert len(s.shape) == 1 and len(ns.shape) == 1
        with th.no_grad():
            s = th.unsqueeze(th.from_numpy(s).float(), dim=0)
            ns = th.unsqueeze(th.from_numpy(ns).float(), dim=0)
            cost = self(self.model_cpu, s, ns)
        reward = -ptu.to_numpy(cost).item()
        return reward

    def cost(self, s: np.ndarray, ns: np.ndarray):
        return -self.reward(s, ns)

    def train_irl(
        self, 
        demo_states: th.Tensor, 
        agent_states: th.Tensor,
        agent_log_probs: th.Tensor
    ):
        """
        Train cost function 
        demo_states, agent_states: (batch_size, T, state_dim)
        agent_log_probs: (batch_size, 1)
        """

        # Compute the first term in IRL loss for demo paths
        # cost_p = self(demo_obs, demo_act)
        s, ns = demo_states[:,:-1], demo_states[:,1:]
        c_demo = th.sum(self(self.model, s, ns), dim=1)
        loss_demo = th.mean(c_demo, dim=0)

        # Compute second term in IRL loss for agent paths
        # cost_q = self(agent_obs, agent_act)
        s, ns = agent_states[:,:-1], agent_states[:,1:]
        c_agent = th.sum(self(self.model, s, ns), dim=1)
        # log_probs = th.sum(agent_log_probs, dim=1, keepdim=True)
        loss_agent = th.logsumexp(-c_agent - agent_log_probs, dim=0)

        loss = loss_demo + loss_agent 

        # # Regularizations
        # if self.lcr_reg_coeff > 0:
        #     loss_lcr = self.lcr_reg_coeff * (self.lcr_reg(cost_p) + self.lcr_reg(cost_q))
        #     loss += loss_lcr
        #     self.logger.record_mean("Disc/LossLCR", loss_lcr.item())

        # if self.gail_reg_coeff > 0:
        #     loss_gail = self.gail_reg_coeff * self.gail_reg(cost_p)
        #     loss += loss_gail
        #     self.logger.record_mean("Disc/LossGAIL", loss_gail.item())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Check gradient norm
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        grad_norms = []
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norms.append(p.grad.detach().data.norm(2))
        grad_norms = th.stack(grad_norms)

        self.optimizer.step()

        # Log metrics
        metrics = {
            'loss': loss,
            'loss_demo': loss_demo,
            'loss_agent': loss_agent,
            'c_demo': c_demo,
            'c_agent': c_agent,
            'agent_log_probs': agent_log_probs,
            'gradient': grad_norms, 
        }
        utils.log_disc_metrics(self.logger, metrics)


        
    ############################################################
    # Regularizations
    ############################################################
    def lcr_reg(self, cost):
        """
        Linearly constant rate regularization that penalizes high frequency variation
        lcr(tau) = sum [(c(x_{t+1}) - c(x_t)) - (c(x_t) - c(x_{t-1}))]^2
        """
        cost_tm1, cost_t, cost_tp1 = cost[:,:-2], cost[:,1:-1], cost[:,2:]
        loss = th.sum(th.square(cost_tm1 - 2*cost_t + cost_tp1))
        return loss

    def gail_reg(self, cost):
        """
        Regularization in GAIL Eq. 13, only penalize expert transitions
        """
        # Ensure cost is negative
        cost -= 1e-6
        assert th.lt(cost, th.zeros_like(cost)).all(), (
            "Cost function not negative everywhere"
        )

        # Numerical stability trick 
        # g(x) = -x - log(1 - e^x) = -x - logsumexp(0, -x)
        zeros = ptu.from_numpy(np.zeros(cost.shape))
        g = -cost - th.logsumexp(th.stack((zeros, cost), dim=-1), dim=-1)
        loss = th.sum(g)
        return loss

    ###############################################
    def copy_model_to_cpu(self) -> None:
        PATH = '/tmp/irl_cost_net.pt'
        th.save(self.model.state_dict(), PATH)
        self.model_cpu.load_state_dict(th.load(PATH, map_location=th.device("cpu")))
