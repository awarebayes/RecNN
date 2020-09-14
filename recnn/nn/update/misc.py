import torch
from recnn import utils
from recnn import data


def temporal_difference(reward, done, gamma, target):
    return reward + (1.0 - done) * gamma * target


def value_update(
    batch,
    params,
    nets,
    optimizer,
    device=torch.device("cpu"),
    debug=None,
    writer=utils.DummyWriter(),
    learn=False,
    step=-1,
):
    """
    Everything is the same as in ddpg_update
    """

    state, action, reward, next_state, done = data.get_base_batch(batch, device=device)

    with torch.no_grad():
        next_action = nets["target_policy_net"](next_state)
        target_value = nets["target_value_net"](next_state, next_action.detach())
        expected_value = temporal_difference(
            reward, done, params["gamma"], target_value
        )
        expected_value = torch.clamp(
            expected_value, params["min_value"], params["max_value"]
        )

    value = nets["value_net"](state, action)

    value_loss = torch.pow(value - expected_value.detach(), 2).mean()

    if learn:
        optimizer["value_optimizer"].zero_grad()
        value_loss.backward(retain_graph=True)
        optimizer["value_optimizer"].step()

    elif not learn:
        debug["next_action"] = next_action
        writer.add_figure(
            "next_action", utils.pairwise_distances_fig(next_action[:50]), step
        )
        writer.add_histogram("value", value, step)
        writer.add_histogram("target_value", target_value, step)
        writer.add_histogram("expected_value", expected_value, step)

    return value_loss
