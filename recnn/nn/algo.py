from recnn.utils import misc
from recnn.nn import update, models
import torch
from torch import nn
import numpy as np


class StateRepresentation:
    pass # TODO


class DDPG:
    def __init__(self, env, params, dataloader, writer, state_encoder='none', device=torch.device('cpu')):
        self.writer = writer
        self.device = device
        self.params = params
        self.env = env
        self.state_encoder = None
        self.dataloader = dataloader
        self.step = 0
        self.init_state_encoder(state_encoder, env)

        self.value_net = models.Critic(env['state_dim'], env['action_dim'], env['hidden_dim'],
                                       params['critic_weight_init']).to(device)
        self.policy_net = models.Actor(env['state_dim'], env['action_dim'], env['hidden_dim'],
                                       params['actor_weight_init']).to(device)

        self.target_value_net = models.Critic(env['state_dim'], env['action_dim'], env['hidden_dim']).to(device)
        self.target_policy_net = models.Actor(env['state_dim'], env['action_dim'], env['hidden_dim']).to(device)

        self.nets = {
            'value_net': self.value_net,
            'target_value_net': self.target_value_net,
            'policy_net': self.policy_net,
            'target_policy_net': self.target_policy_net,
        }

        self.optimizer = None

    """
    You need to manually register optimizers you like
    ddpg = DDPG(...)
    state, policy, value = ddpg.get_parameters()
    value_optimizer = optim.RAdam(value, lr=params['value_lr'], weight_decay=1e-2)
    policy_optimizer = optim.RAdam(state + policy, lr=params['policy_lr'], weight_decay=1e-2)
    optimizer = {
        'policy_optimizer': policy_optimizer,
        'value_optimizer':  value_optimizer
    }
    ddpg.register_optimizers(optimizer)
    """

    def register_optimizers(self, optimizer):
        self.optimizer = optimizer
