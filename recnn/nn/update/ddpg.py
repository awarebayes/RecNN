import torch
from recnn import utils
from recnn import data
from recnn.utils import soft_update
from recnn.nn.update import value_update


def ddpg_update(
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
    :param batch: batch [state, action, reward, next_state] returned by environment.
    :param params: dict of algorithm parameters.
    :param nets: dict of networks.
    :param optimizer: dict of optimizers
    :param device: torch.device
    :param debug: dictionary where debug data about actions is saved
    :param writer: torch.SummaryWriter
    :param learn: whether to learn on this step (used for testing)
    :param step: integer step for policy update
    :return: loss dictionary

    How parameters should look like::

        params = {
            'gamma'      : 0.99,
            'min_value'  : -10,
            'max_value'  : 10,
            'policy_step': 3,
            'soft_tau'   : 0.001,
            'policy_lr'  : 1e-5,
            'value_lr'   : 1e-5,
            'actor_weight_init': 3e-1,
            'critic_weight_init': 6e-1,
        }
        nets = {
            'value_net': models.Critic,
            'target_value_net': models.Critic,
            'policy_net': models.Actor,
            'target_policy_net': models.Actor,
        }
        optimizer - {
            'policy_optimizer': some optimizer
            'value_optimizer':  some optimizer
        }

    """

    state, action, reward, next_state, _ = data.get_base_batch(batch, device=device)

    # --------------------------------------------------------#
    # Value Learning

    value_loss = value_update(
        batch,
        params,
        nets,
        optimizer,
        writer=writer,
        device=device,
        debug=debug,
        learn=learn,
        step=step,
    )

    # --------------------------------------------------------#
    # Policy learning

    gen_action = nets["policy_net"](state)
    policy_loss = -nets["value_net"](state, gen_action)

    if not learn:
        debug["gen_action"] = gen_action
        writer.add_histogram("policy_loss", policy_loss, step)
        writer.add_figure(
            "next_action", utils.pairwise_distances_fig(gen_action[:50]), step
        )
    policy_loss = policy_loss.mean()

    if learn and step % params["policy_step"] == 0:
        optimizer["policy_optimizer"].zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(nets["policy_net"].parameters(), -1, 1)
        optimizer["policy_optimizer"].step()

        soft_update(
            nets["value_net"], nets["target_value_net"], soft_tau=params["soft_tau"]
        )
        soft_update(
            nets["policy_net"], nets["target_policy_net"], soft_tau=params["soft_tau"]
        )

    losses = {"value": value_loss.item(), "policy": policy_loss.item(), "step": step}
    utils.write_losses(writer, losses, kind="train" if learn else "test")
    return losses
