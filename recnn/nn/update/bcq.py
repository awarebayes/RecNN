import torch
import torch.functional as F

from recnn import utils
from recnn import data
from recnn.utils import soft_update
from recnn.nn.update import temporal_difference


# batch, params, writer, debug, learn=True, step=-1
def bcq_update(
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
            # algorithm parameters
            'gamma'              : 0.99,
            'soft_tau'           : 0.001,
            'n_generator_samples': 10,
            'perturbator_step'   : 30,

            # learning rates
            'perturbator_lr' : 1e-5,
            'value_lr'       : 1e-5,
            'generator_lr'   : 1e-3,
        }


        nets = {
            'generator_net': models.bcqGenerator,
            'perturbator_net': models.bcqPerturbator,
            'target_perturbator_net': models.bcqPerturbator,
            'value_net1': models.Critic,
            'target_value_net1': models.Critic,
            'value_net2': models.Critic,
            'target_value_net2': models.Critic,
        }

        optimizer = {
            'generator_optimizer': some optimizer
            'policy_optimizer': some optimizer
            'value_optimizer1':  some optimizer
            'value_optimizer2':  some optimizer
        }


    """

    if debug is None:
        debug = dict()
    state, action, reward, next_state, done = data.get_base_batch(batch, device=device)
    batch_size = done.size(0)

    # --------------------------------------------------------#
    # Variational Auto-Encoder Learning
    recon, mean, std = nets["generator_net"](state, action)
    recon_loss = F.mse_loss(recon, action)
    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    generator_loss = recon_loss + 0.5 * KL_loss

    if not learn:
        writer.add_histogram("generator_mean", mean, step)
        writer.add_histogram("generator_std", std, step)
        debug["recon"] = recon
        writer.add_figure(
            "reconstructed", utils.pairwise_distances_fig(recon[:50]), step
        )

    if learn:
        optimizer["generator_optimizer"].zero_grad()
        generator_loss.backward()
        optimizer["generator_optimizer"].step()
    # --------------------------------------------------------#
    # Value Learning
    with torch.no_grad():
        # p.s. repeat_interleave was added in torch 1.1
        # if an error pops up, run 'conda update pytorch'
        state_rep = torch.repeat_interleave(
            next_state, params["n_generator_samples"], 0
        )
        sampled_action = nets["generator_net"].decode(state_rep)
        perturbed_action = nets["target_perturbator_net"](state_rep, sampled_action)
        target_Q1 = nets["target_value_net1"](state_rep, perturbed_action)
        target_Q2 = nets["target_value_net1"](state_rep, perturbed_action)
        target_value = 0.75 * torch.min(target_Q1, target_Q2)  # value soft update
        target_value += 0.25 * torch.max(target_Q1, target_Q2)  #
        target_value = target_value.view(batch_size, -1).max(1)[0].view(-1, 1)

        expected_value = temporal_difference(
            reward, done, params["gamma"], target_value
        )

    value = nets["value_net1"](state, action)
    value_loss = torch.pow(value - expected_value.detach(), 2).mean()

    if learn:
        optimizer["value_optimizer1"].zero_grad()
        optimizer["value_optimizer2"].zero_grad()
        value_loss.backward()
        optimizer["value_optimizer1"].step()
        optimizer["value_optimizer2"].step()
    else:
        writer.add_histogram("value", value, step)
        writer.add_histogram("target_value", target_value, step)
        writer.add_histogram("expected_value", expected_value, step)
        writer.close()

    # --------------------------------------------------------#
    # Perturbator learning
    sampled_actions = nets["generator_net"].decode(state)
    perturbed_actions = nets["perturbator_net"](state, sampled_actions)
    perturbator_loss = -nets["value_net1"](state, perturbed_actions)
    if not learn:
        writer.add_histogram("perturbator_loss", perturbator_loss, step)
    perturbator_loss = perturbator_loss.mean()

    if learn:
        if step % params["perturbator_step"] == 0:
            optimizer["perturbator_optimizer"].zero_grad()
            perturbator_loss.backward()
            torch.nn.utils.clip_grad_norm_(nets["perturbator_net"].parameters(), -1, 1)
            optimizer["perturbator_optimizer"].step()

        soft_update(
            nets["value_net1"], nets["target_value_net1"], soft_tau=params["soft_tau"]
        )
        soft_update(
            nets["value_net2"], nets["target_value_net2"], soft_tau=params["soft_tau"]
        )
        soft_update(
            nets["perturbator_net"],
            nets["target_perturbator_net"],
            soft_tau=params["soft_tau"],
        )
    else:
        debug["sampled_actions"] = sampled_actions
        debug["perturbed_actions"] = perturbed_actions
        writer.add_figure(
            "sampled_actions", utils.pairwise_distances_fig(sampled_actions[:50]), step
        )
        writer.add_figure(
            "perturbed_actions",
            utils.pairwise_distances_fig(perturbed_actions[:50]),
            step,
        )

    # --------------------------------------------------------#

    losses = {
        "value": value_loss.item(),
        "perturbator": perturbator_loss.item(),
        "generator": generator_loss.item(),
        "step": step,
    }

    utils.write_losses(writer, losses, kind="train" if learn else "test")
    return losses
