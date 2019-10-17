import torch
from recnn import utils
from recnn import data

"""
helper function for weight update
"""


def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


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

    state, action, reward, next_state = data.get_base_batch(batch, done=False)

    # --------------------------------------------------------#
    # Value Learning

    with torch.no_grad():
        next_action = nets['target_policy_net'](next_state)
        target_value = nets['target_value_net'](next_state, next_action.detach())
        expected_value = reward + params['gamma'] * target_value
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

    batch = [i.to(device) for i in batch]
    state, action, reward, next_state, done = batch

    # --------------------------------------------------------#
    # Value Learning

    reward = reward.unsqueeze(1)
    done = done.unsqueeze(1)

    next_action = nets['target_policy_net'](next_state)
    noise = torch.normal(torch.zeros(next_action.size()),
                         params['noise_std']).to(device)
    noise = torch.clamp(noise, -params['noise_clip'], params['noise_clip'])
    next_action += noise

    with torch.no_grad():
        target_q_value1 = nets['target_value_net1'](next_state, next_action)
        target_q_value2 = nets['target_value_net2'](next_state, next_action)
        target_q_value = torch.min(target_q_value1, target_q_value2)
        expected_q_value = reward + (1.0 - done) * params['gamma'] * target_q_value

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


