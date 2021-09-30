import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.training = training
        self.nn_baseline = nn_baseline

        # Continuous action
        self.logits_na = None
        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers,
            size=self.size
        )
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.mean_net.to(ptu.device)
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )


    # sample action(s) and corresponding log prob(s)
    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation_tensor = torch.tensor(observation, dtype=torch.float).to(ptu.device)
        action_distribution = self.forward(observation_tensor)
        actions = action_distribution.sample() 
        log_probs = action_distribution.log_prob(actions)

        return ptu.to_numpy(actions), ptu.to_numpy(log_probs)


    # update/train this policy
    def update(self, observations, actions, **kwargs):
        """
        """
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)


        actions_distribution = self.forward(observations)
        log_probs = actions_distribution.log_prob(actions)


        # TODO: optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }

    # Returning a torch.distributions.Distribution objects avoids the 
    # reparameterization trick problem
    def forward(self, observation: torch.Tensor) -> distributions.Distribution:
        return distributions.Normal(
            self.mean_net(observation),
            torch.exp(self.logstd)[None],
        )

