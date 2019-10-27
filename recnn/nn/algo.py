from .. import utils, optim
from . import update

import torch
import copy


class DDPG:
    def __init__(self, policy_net, value_net):
        # these are target networks that we need for ddpg algorigm to work
        target_policy_net = copy.deepcopy(policy_net)
        target_value_net = copy.deepcopy(value_net)

        target_policy_net.eval()
        target_value_net.eval()

        # soft update
        utils.soft_update(value_net, target_value_net, soft_tau=1.0)
        utils.soft_update(policy_net, target_policy_net, soft_tau=1.0)

        # define optimizers
        value_optimizer = optim.Ranger(value_net.parameters(), lr=1e-5, weight_decay=1e-2)
        policy_optimizer = optim.Ranger(policy_net.parameters(), lr=1e-5, weight_decay=1e-2)

        self.nets = {
            'value_net': value_net,
            'target_value_net': target_value_net,
            'policy_net': policy_net,
            'target_policy_net': target_policy_net,
        }

        self.optimizers = {
            'policy_optimizer': policy_optimizer,
            'value_optimizer': value_optimizer
        }

        self.params = {
            'gamma': 0.99,
            'min_value': -10,
            'max_value': 10,
            'policy_step': 10,
            'soft_tau': 0.001,
        }

        self._step = 0

        self.debug = {}

        # by default it will not output anything
        # use torch.SummaryWriter instance if you want output
        self.writer = utils.misc.DummyWriter()

        self.device = torch.device('cpu')

        self.loss_layout = {
            'test': {'value': [], 'policy': [], 'step': []},
            'train': {'value': [], 'policy': [], 'step': []}
        }

    def to(self, device):
        self.nets = {k: v.to(device) for k, v in self.nets.items()}
        self.device = device
        return self

    def update(self, batch, learn):
        return update.ddpg_update(batch, self.params, self.nets, self.optimizers,
                                  self.device, self.debug, self.writer, step=self._step, learn=learn)

    def step(self):
        self._step += 1


