import torch

# == recnn ==
import recnn
import torch_optimizer as optim

state = torch.randn(10, 1290)
action = torch.randn(10, 128)
reward = torch.randn(10, 1)
next_state = torch.randn(10, 1290)
done = torch.randn(10, 1)
batch = {
    "state": state,
    "action": action,
    "reward": reward,
    "next_state": next_state,
    "done": done,
}

value_net = recnn.nn.Critic(1290, 128, 256, 54e-2)
policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)


def test_recommendation():

    recommendation = policy_net(state)
    value = value_net(state, recommendation)

    assert recommendation.std() > 0 and recommendation.mean != 0
    assert value.std() > 0


def check_loss_and_networks(loss, nets):
    assert loss["value"] > 0 and loss["policy"] != 0 and loss["step"] == 0
    for name, netw in nets.items():
        assert netw.training == ("target" not in name)


def test_update_function():
    target_value_net = recnn.nn.Critic(1290, 128, 256)
    target_policy_net = recnn.nn.Actor(1290, 128, 256)

    target_policy_net.eval()
    target_value_net.eval()

    # soft update
    recnn.utils.soft_update(value_net, target_value_net, soft_tau=1.0)
    recnn.utils.soft_update(policy_net, target_policy_net, soft_tau=1.0)

    # define optimizers
    value_optimizer = optim.RAdam(value_net.parameters(), lr=1e-5, weight_decay=1e-2)
    policy_optimizer = optim.RAdam(policy_net.parameters(), lr=1e-5, weight_decay=1e-2)

    nets = {
        "value_net": value_net,
        "target_value_net": target_value_net,
        "policy_net": policy_net,
        "target_policy_net": target_policy_net,
    }

    optimizer = {
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }

    debug = {}
    writer = recnn.utils.misc.DummyWriter()

    step = 0
    params = {
        "gamma": 0.99,
        "min_value": -10,
        "max_value": 10,
        "policy_step": 10,
        "soft_tau": 0.001,
    }

    loss = recnn.nn.update.ddpg_update(
        batch, params, nets, optimizer, torch.device("cpu"), debug, writer, step=step
    )

    check_loss_and_networks(loss, nets)


def test_algo():
    value_net = recnn.nn.Critic(1290, 128, 256, 54e-2)
    policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)

    ddpg = recnn.nn.DDPG(policy_net, value_net)
    ddpg = ddpg
    loss = ddpg.update(batch, learn=True)
    check_loss_and_networks(loss, ddpg.nets)
