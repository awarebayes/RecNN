import torch
import torch.functional as F
from recnn import utils
from recnn import data
from recnn.utils import soft_update


def temporal_difference(reward, done, gamma, target):
    return reward + (1.0 - done) * gamma * target

def ddpg_update(batch, params, nets, optimizer, device, debug, writer=False, learn=True, step=-1):

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

    state, action, reward, next_state, done = data.get_base_batch(batch, device=device)

    # --------------------------------------------------------#
    # Value Learning

    with torch.no_grad():
        next_action = nets['target_policy_net'](next_state)
        target_value = nets['target_value_net'](next_state, next_action.detach())
        expected_value = temporal_difference(reward, done, params['gamma'], target_value)
        expected_value = torch.clamp(expected_value,
                                     params['min_value'], params['max_value'])

    value = nets['value_net'](state, action)

    value_loss = torch.pow(value - expected_value.detach(), 2).mean()

    if learn:
        optimizer['value_optimizer'].zero_grad()
        value_loss.backward(retain_graph=True)
        optimizer['value_optimizer'].step()

    elif not learn:
        debug['next_action'] = next_action
        writer.add_figure('next_action',
                          utils.pairwise_distances_fig(next_action[:50]), step)
        writer.add_histogram('value', value, step)
        writer.add_histogram('target_value', target_value, step)
        writer.add_histogram('expected_value', expected_value, step)

    # --------------------------------------------------------#
    # Policy learning

    gen_action = nets['policy_net'](state)
    policy_loss = -nets['value_net'](state, gen_action)

    if not learn:
        debug['gen_action'] = gen_action
        writer.add_histogram('policy_loss', policy_loss, step)
        writer.add_figure('next_action',
                          utils.pairwise_distances_fig(gen_action[:50]), step)
    policy_loss = policy_loss.mean()

    if learn and step % params['policy_step'] == 0:
        optimizer['policy_optimizer'].zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(nets['policy_net'].parameters(), -1, 1)
        optimizer['policy_optimizer'].step()

        soft_update(nets['value_net'], nets['target_value_net'], soft_tau=params['soft_tau'])
        soft_update(nets['policy_net'], nets['target_policy_net'], soft_tau=params['soft_tau'])

    losses = {'value': value_loss.item(), 'policy': policy_loss.item(), 'step': step}
    utils.write_losses(writer, losses, kind='train' if learn else 'test')
    return losses


def td3_update(batch, params, nets, optimizer, writer, device, debug, learn=True, step=-1):
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


        nets = {
            'value_net1': models.Critic,
            'target_value_net1': models.Critic,
            'value_net2': models.Critic,
            'target_value_net2': models.Critic,
            'policy_net': models.Actor,
            'target_policy_net': models.Actor,
        }

        optimizer = {
            'policy_optimizer': some optimizer
            'value_optimizer1':  some optimizer
            'value_optimizer2':  some optimizer
        }


    """

    state, action, reward, next_state, done = data.get_base_batch(batch, device=device)

    # --------------------------------------------------------#
    # Value Learning

    next_action = nets['target_policy_net'](next_state)
    noise = torch.normal(torch.zeros(next_action.size()),
                         params['noise_std']).to(device)
    noise = torch.clamp(noise, -params['noise_clip'], params['noise_clip'])
    next_action += noise

    with torch.no_grad():
        target_q_value1 = nets['target_value_net1'](next_state, next_action)
        target_q_value2 = nets['target_value_net2'](next_state, next_action)
        target_q_value = torch.min(target_q_value1, target_q_value2)
        expected_q_value = temporal_difference(reward, done, params['gamma'], target_q_value)

    q_value1 = nets['value_net1'](state, action)
    q_value2 = nets['value_net2'](state, action)

    value_criterion = torch.nn.MSELoss()
    value_loss1 = value_criterion(q_value1, expected_q_value.detach())
    value_loss2 = value_criterion(q_value2, expected_q_value.detach())

    if learn:
        optimizer['value_optimizer1'].zero_grad()
        value_loss1.backward()
        optimizer['value_optimizer1'].step()

        optimizer['value_optimizer2'].zero_grad()
        value_loss2.backward()
        optimizer['value_optimizer2'].step()
    else:
        debug['next_action'] = next_action
        writer.add_figure('next_action',
                          utils.pairwise_distances_fig(next_action[:50]), step)
        writer.add_histogram('value1', q_value1, step)
        writer.add_histogram('value2', q_value2, step)
        writer.add_histogram('target_value', target_q_value, step)
        writer.add_histogram('expected_value', expected_q_value, step)

    # --------------------------------------------------------#
    # Policy learning

    gen_action = nets['policy_net'](state)
    policy_loss = nets['value_net1'](state, gen_action)
    policy_loss = -policy_loss

    if not learn:
        debug['gen_action'] = gen_action
        writer.add_figure('gen_action',
                          utils.pairwise_distances_fig(gen_action[:50]), step)
        writer.add_histogram('policy_loss', policy_loss, step)

    policy_loss = policy_loss.mean()

    # delayed policy update
    if step % params['policy_update'] == 0 and learn:
        optimizer['policy_optimizer'].zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(nets['policy_net'].parameters(), -1, 1)
        optimizer['policy_optimizer'].step()

        soft_update(nets['value_net1'], nets['target_value_net1'], soft_tau=params['soft_tau'])
        soft_update(nets['value_net2'], nets['target_value_net2'], soft_tau=params['soft_tau'])

    losses = {'value1': value_loss1.item(),
              'value2': value_loss2.item(),
              'policy': policy_loss.item(),
              'step': step}
    utils.write_losses(writer, losses, kind='train' if learn else 'test')
    return losses


# batch, params, writer, debug, learn=True, step=-1
def bcq_update(batch, params, nets, optimizer, writer, device, debug, learn=True, step=-1):

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

    state, action, reward, next_state, done = data.get_base_batch(batch, device=device)
    batch_size = done.size(0)

    # --------------------------------------------------------#
    # Variational Auto-Encoder Learning
    recon, mean, std = nets['generator_net'](state, action)
    recon_loss = F.mse_loss(recon, action)
    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    generator_loss = recon_loss + 0.5 * KL_loss

    if not learn:
        writer.add_histogram('generator_mean', mean, step)
        writer.add_histogram('generator_std', std, step)
        debug['recon'] = recon
        writer.add_figure('reconstructed',
                          utils.pairwise_distances_fig(recon[:50]), step)

    if learn:
        optimizer['generator_optimizer'].zero_grad()
        generator_loss.backward()
        optimizer['generator_optimizer'].step()
    # --------------------------------------------------------#
    # Value Learning
    with torch.no_grad():
        # p.s. repeat_interleave was added in torch 1.1
        # if an error pops up, run 'conda update pytorch'
        state_rep = torch.repeat_interleave(next_state, params['n_generator_samples'], 0)
        sampled_action = nets['generator_net'].decode(state_rep)
        perturbed_action = nets['target_perturbator_net'](state_rep, sampled_action)
        target_Q1 = nets['target_value_net1'](state_rep, perturbed_action)
        target_Q2 = nets['target_value_net1'](state_rep, perturbed_action)
        target_value = 0.75 * torch.min(target_Q1, target_Q2)  # value soft update
        target_value += 0.25 * torch.max(target_Q1, target_Q2)  #
        target_value = target_value.view(batch_size, -1).max(1)[0].view(-1, 1)

        expected_value = temporal_difference(reward, done, params['gamma'], target_value)

    value = nets['value_net1'](state, action)
    value_loss = torch.pow(value - expected_value.detach(), 2).mean()

    if learn:
        optimizer['value_optimizer1'].zero_grad()
        optimizer['value_optimizer2'].zero_grad()
        value_loss.backward()
        optimizer['value_optimizer1'].step()
        optimizer['value_optimizer2'].step()
    else:
        writer.add_histogram('value', value, step)
        writer.add_histogram('target_value', target_value, step)
        writer.add_histogram('expected_value', expected_value, step)
        writer.close()

    # --------------------------------------------------------#
    # Perturbator learning
    sampled_actions = nets['generator_net'].decode(state)
    perturbed_actions = nets['perturbator_net'](state, sampled_actions)
    perturbator_loss = -nets['value_net1'](state, perturbed_actions)
    if not learn:
        writer.add_histogram('perturbator_loss', perturbator_loss, step)
    perturbator_loss = perturbator_loss.mean()

    if learn:
        if step % params['perturbator_step']:
            optimizer['perturbator_optimizer'].zero_grad()
            perturbator_loss.backward()
            torch.nn.utils.clip_grad_norm_(nets['perturbator_net'].parameters(), -1, 1)
            optimizer['perturbator_optimizer'].step()

        soft_update(nets['value_net1'], nets['target_value_net1'], soft_tau=params['soft_tau'])
        soft_update(nets['value_net2'], nets['target_value_net2'], soft_tau=params['soft_tau'])
        soft_update(nets['perturbator_net'], nets['target_perturbator_net'], soft_tau=params['soft_tau'])
    else:
        debug['sampled_actions'] = sampled_actions
        debug['perturbed_actions'] = perturbed_actions
        writer.add_figure('sampled_actions',
                          utils.pairwise_distances_fig(sampled_actions[:50]), step)
        writer.add_figure('perturbed_actions',
                          utils.pairwise_distances_fig(perturbed_actions[:50]), step)

    # --------------------------------------------------------#

    losses = {'value': value_loss.item(),
              'perturbator': perturbator_loss.item(),
              'generator': generator_loss.item(),
              'step': step}

    utils.write_losses(writer, losses, kind='train' if learn else 'test')
    return losses
