from recnn import utils
from recnn.nn import update
from recnn.nn.update import ChooseREINFORCE

import torch
import torch_optimizer as optim
import copy


"""
 Algorithms as is aren't much. Just classes with pre set up parameters, optimizers and stuff.
"""


class Algo:
    def __init__(self):
        self.nets = {
            "value_net": None,
            "policy_net": None,
        }

        self.optimizers = {"policy_optimizer": None, "value_optimizer": None}

        self.params = {"Some parameters here": None}

        self._step = 0

        self.debug = {}

        # by default it will not output anything
        # use torch.SummaryWriter instance if you want output
        self.writer = utils.misc.DummyWriter()

        self.device = torch.device("cpu")

        self.loss_layout = {
            "test": {"value": [], "policy": [], "step": []},
            "train": {"value": [], "policy": [], "step": []},
        }

        self.algorithm = None

    def update(self, batch, learn=True):
        return self.algorithm(
            batch,
            self.params,
            self.nets,
            self.optimizers,
            device=self.device,
            debug=self.debug,
            writer=self.writer,
            learn=learn,
            step=self._step,
        )

    def to(self, device):
        self.nets = {k: v.to(device) for k, v in self.nets.items()}
        self.device = device
        return self

    def step(self):
        self._step += 1


class DDPG(Algo):
    def __init__(self, policy_net, value_net):

        super(DDPG, self).__init__()

        self.algorithm = update.ddpg_update

        # these are target networks that we need for ddpg algorigm to work
        target_policy_net = copy.deepcopy(policy_net)
        target_value_net = copy.deepcopy(value_net)

        target_policy_net.eval()
        target_value_net.eval()

        # soft update
        utils.soft_update(value_net, target_value_net, soft_tau=1.0)
        utils.soft_update(policy_net, target_policy_net, soft_tau=1.0)

        # define optimizers
        value_optimizer = optim.Ranger(
            value_net.parameters(), lr=1e-5, weight_decay=1e-2
        )
        policy_optimizer = optim.Ranger(
            policy_net.parameters(), lr=1e-5, weight_decay=1e-2
        )

        self.nets = {
            "value_net": value_net,
            "target_value_net": target_value_net,
            "policy_net": policy_net,
            "target_policy_net": target_policy_net,
        }

        self.optimizers = {
            "policy_optimizer": policy_optimizer,
            "value_optimizer": value_optimizer,
        }

        self.params = {
            "gamma": 0.99,
            "min_value": -10,
            "max_value": 10,
            "policy_step": 10,
            "soft_tau": 0.001,
        }

        self.loss_layout = {
            "test": {"value": [], "policy": [], "step": []},
            "train": {"value": [], "policy": [], "step": []},
        }


class TD3(Algo):
    def __init__(self, policy_net, value_net1, value_net2):

        super(TD3, self).__init__()

        self.algorithm = update.td3_update

        # these are target networks that we need for TD3 algorigm to work
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
        value_optimizer1 = optim.Ranger(
            value_net1.parameters(), lr=1e-5, weight_decay=1e-2
        )
        value_optimizer2 = optim.Ranger(
            value_net2.parameters(), lr=1e-5, weight_decay=1e-2
        )
        policy_optimizer = optim.Ranger(
            policy_net.parameters(), lr=1e-5, weight_decay=1e-2
        )

        self.nets = {
            "value_net1": value_net1,
            "target_value_net1": target_value_net1,
            "value_net2": value_net2,
            "target_value_net2": target_value_net2,
            "policy_net": policy_net,
            "target_policy_net": target_policy_net,
        }

        self.optimizers = {
            "policy_optimizer": policy_optimizer,
            "value_optimizer1": value_optimizer1,
            "value_optimizer2": value_optimizer2,
        }

        self.params = {
            "gamma": 0.99,
            "noise_std": 0.5,
            "noise_clip": 3,
            "soft_tau": 0.001,
            "policy_update": 10,
            "policy_lr": 1e-5,
            "value_lr": 1e-5,
            "actor_weight_init": 25e-2,
            "critic_weight_init": 6e-1,
        }

        self.loss_layout = {
            "test": {"value1": [], "value2": [], "policy": [], "step": []},
            "train": {"value1": [], "value2": [], "policy": [], "step": []},
        }


class Reinforce(Algo):
    def __init__(self, policy_net, value_net):

        super(Reinforce, self).__init__()

        self.algorithm = update.reinforce_update

        # these are target networks that we need for ddpg algorigm to work
        target_policy_net = copy.deepcopy(policy_net)
        target_value_net = copy.deepcopy(value_net)

        target_policy_net.eval()
        target_value_net.eval()

        # soft update
        utils.soft_update(value_net, target_value_net, soft_tau=1.0)
        utils.soft_update(policy_net, target_policy_net, soft_tau=1.0)

        # define optimizers
        value_optimizer = optim.Ranger(
            value_net.parameters(), lr=1e-5, weight_decay=1e-2
        )
        policy_optimizer = optim.Ranger(
            policy_net.parameters(), lr=1e-5, weight_decay=1e-2
        )

        self.nets = {
            "value_net": value_net,
            "target_value_net": target_value_net,
            "policy_net": policy_net,
            "target_policy_net": target_policy_net,
        }

        self.optimizers = {
            "policy_optimizer": policy_optimizer,
            "value_optimizer": value_optimizer,
        }

        self.params = {
            "reinforce": ChooseREINFORCE(ChooseREINFORCE.basic_reinforce),
            "K": 10,
            "gamma": 0.99,
            "min_value": -10,
            "max_value": 10,
            "policy_step": 10,
            "soft_tau": 0.001,
        }

        self.loss_layout = {
            "test": {"value": [], "policy": [], "step": []},
            "train": {"value": [], "policy": [], "step": []},
        }
