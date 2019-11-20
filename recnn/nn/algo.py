from .. import utils, optim
from . import update

import torch
import copy


class DDPG:

    """
    Whats inside::

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

    """

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
                                  device=self.device,
                                  debug=self.debug, writer=self.writer,
                                  learn=learn, step=self._step)

    def step(self):
        self._step += 1


class TD3:

    """

        What's inside::

            def __init__(self, policy_net, value_net1, value_net2):
                # these are target networks that we need for ddpg algorigm to work
                target_policy_net = copy.deepcopy(policy_net)
                target_value_net1 = copy.deepcopy(value_net1)
                target_value_net2 = copy.deepcopy(value_net2)

                target_policy_net.eval()
                target_value_net1.eval()
                target_value_net2.eval()

                # soft update
                utils.soft_update(value_net1, target_value_net1, soft_tau=1.0)
                utils.soft_update(value_net2, target_value_net2, soft_tau=1.0)
                utils.soft_update(policy_net, target_policy_net, soft_tau=1.0)

                # define optimizers
                value_optimizer1 = optim.Ranger(value_net1.parameters(), lr=1e-5, weight_decay=1e-2)
                value_optimizer2 = optim.Ranger(value_net2.parameters(), lr=1e-5, weight_decay=1e-2)
                policy_optimizer = optim.Ranger(policy_net.parameters(), lr=1e-5, weight_decay=1e-2)

                self.nets = {
                    'value_net1': value_net1,
                    'target_value_net1': target_value_net1,
                    'value_net2': value_net2,
                    'target_value_net2': target_value_net2,
                    'policy_net': policy_net,
                    'target_policy_net': target_policy_net,
                }

                self.optimizers = {
                    'policy_optimizer': policy_optimizer,
                    'value_optimizer1': value_optimizer1,
                    'value_optimizer2': value_optimizer2,
                }

                self.params = {
                    'gamma': 0.99,
                    'noise_std': 0.5,
                    'noise_clip': 3,
                    'soft_tau': 0.001,
                    'policy_update': 10,

                    'policy_lr': 1e-5,
                    'value_lr': 1e-5,

                    'actor_weight_init': 25e-2,
                    'critic_weight_init': 6e-1,
                }

                self._step = 0

                self.debug = {}

                # by default it will not output anything
                # use torch.SummaryWriter instance if you want output
                self.writer = utils.misc.DummyWriter()

                self.device = torch.device('cpu')

                self.loss_layout = {
                    'test': {'value1': [], 'value2': [], 'policy': [], 'step': []},
                    'train': {'value1': [], 'value2': [], 'policy': [], 'step': []}
                }
    """

    def __init__(self, policy_net, value_net1, value_net2):
        # these are target networks that we need for ddpg algorigm to work
        target_policy_net = copy.deepcopy(policy_net)
        target_value_net1 = copy.deepcopy(value_net1)
        target_value_net2 = copy.deepcopy(value_net2)

        target_policy_net.eval()
        target_value_net1.eval()
        target_value_net2.eval()

        # soft update
        utils.soft_update(value_net1, target_value_net1, soft_tau=1.0)
        utils.soft_update(value_net2, target_value_net2, soft_tau=1.0)
        utils.soft_update(policy_net, target_policy_net, soft_tau=1.0)

        # define optimizers
        value_optimizer1 = optim.Ranger(value_net1.parameters(), lr=1e-5, weight_decay=1e-2)
        value_optimizer2 = optim.Ranger(value_net2.parameters(), lr=1e-5, weight_decay=1e-2)
        policy_optimizer = optim.Ranger(policy_net.parameters(), lr=1e-5, weight_decay=1e-2)

        self.nets = {
            'value_net1': value_net1,
            'target_value_net1': target_value_net1,
            'value_net2': value_net2,
            'target_value_net2': target_value_net2,
            'policy_net': policy_net,
            'target_policy_net': target_policy_net,
        }

        self.optimizers = {
            'policy_optimizer': policy_optimizer,
            'value_optimizer1': value_optimizer1,
            'value_optimizer2': value_optimizer2,
        }

        self.params = {
            'gamma': 0.99,
            'noise_std': 0.5,
            'noise_clip': 3,
            'soft_tau': 0.001,
            'policy_update': 10,

            'policy_lr': 1e-5,
            'value_lr': 1e-5,

            'actor_weight_init': 25e-2,
            'critic_weight_init': 6e-1,
        }

        self._step = 0

        self.debug = {}

        # by default it will not output anything
        # use torch.SummaryWriter instance if you want output
        self.writer = utils.misc.DummyWriter()

        self.device = torch.device('cpu')

        self.loss_layout = {
            'test': {'value1': [], 'value2': [], 'policy': [], 'step': []},
            'train': {'value1': [], 'value2': [], 'policy': [], 'step': []}
        }

    def to(self, device):
        self.nets = {k: v.to(device) for k, v in self.nets.items()}
        self.device = device
        return self

    def update(self, batch, learn):
        return update.td3_update(batch, self.params, self.nets, self.optimizers,
                                 device=self.device,
                                 debug=self.debug, writer=self.writer,
                                 learn=learn, step=self._step)

    def step(self):
        self._step += 1



