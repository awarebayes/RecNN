import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from . import plot

"""
helper function for weight update
"""


def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


"""
batch - batch returned by DataLoader(data.UserDataset)
params - {
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

nets - {
    'value_net': models.Critic,
    'target_value_net': models.Critic,
    'policy_net': models.Actor,
    'target_policy_net': models.Actor,
}

optimizer - {
    'policy_optimizer': some optimizer 
    'value_optimizer':  some optimizer
}

device - torch.device (cpu/cuda)
debugger - recnn.debugger instance for pretty plotting
learn - whether to learn on this step (used for testing)

"""


def ddpg_update(batch, params, nets, optimizer, device, debugger=False, learn=True, step=-1):
    batch = [i.to(device) for i in batch]
    state, action, reward, next_state, done = batch
    reward = reward.unsqueeze(1)
    # done = done.unsqueeze(1)

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

    elif not learn and debugger:
            debugger.writer.add_figure('next_action',
                                        plot.pairwise_distances_fig(next_action[:50]), step)
            debugger.writer.add_histogram('value', value, step)
            debugger.writer.add_histogram('target_value', target_value, step)
            debugger.writer.add_histogram('expected_value', expected_value, step)

    # --------------------------------------------------------#
    # Policy learning

    gen_action = nets['policy_net'](state)
    policy_loss = -nets['value_net'](state, gen_action)

    if not learn and debugger:
        debugger.log_object('gen_action', gen_action, test=(not learn))
        debugger.writer.add_histogram('policy_loss', policy_loss, step)
        debugger.writer.add_figure('next_action',
                          plot.pairwise_distances_fig(gen_action[:50]), step)
    policy_loss = policy_loss.mean()

    if learn and step % params['policy_step'] == 0:
        optimizer['policy_optimizer'].zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(nets['policy_net'].parameters(), -1, 1)
        optimizer['policy_optimizer'].step()

        soft_update(nets['value_net'], nets['target_value_net'], soft_tau=params['soft_tau'])
        soft_update(nets['policy_net'], nets['target_policy_net'], soft_tau=params['soft_tau'])

    losses = {'value': value_loss.item(), 'policy': policy_loss.item(), 'step': step}
    return losses

