from . import learning, models, optim, misc
import torch
from torch import nn
import numpy as np


class StateRepresentation:
    pass # TODO


class DDPG:
    def __init__(self, env, params, dataloader, writer, state_encoder='none', device=torch.device('cpu')):
        self.writer = writer
        self.device = device
        self.params = params
        self.env = env
        self.state_encoder = None
        self.dataloader = dataloader
        self.step = 0
        self.init_state_encoder(state_encoder, env)

        self.value_net = models.Critic(env['state_dim'], env['action_dim'], env['hidden_dim'],
                                       params['critic_weight_init']).to(device)
        self.policy_net = models.Actor(env['state_dim'], env['action_dim'], env['hidden_dim'],
                                       params['actor_weight_init']).to(device)

        self.target_value_net = models.Critic(env['state_dim'], env['action_dim'], env['hidden_dim']).to(device)
        self.target_policy_net = models.Actor(env['state_dim'], env['action_dim'], env['hidden_dim']).to(device)

        self.nets = {
            'value_net': self.value_net,
            'target_value_net': self.target_value_net,
            'policy_net': self.policy_net,
            'target_policy_net': self.target_policy_net,
        }

        self.optimizer = None

    def test(self):
        max_buf_size = 10000
        buffer = misc.ReplayBuffer(max_buf_size)

        while buffer.len() <= max_buf_size:
            batch = next(iter(self.dataloder['test']))
            batch = [i.to(self.device) for i in batch]
            items, ratings, sizes = batch
            hidden = None
            state = None
            for t in range(int(sizes.min().item()) - 1):
                action = items[:, t]
                reward = ratings[:, t].unsqueeze(-1)
                s = torch.cat([action, reward], 1).unsqueeze(0)
                next_state, hidden = self.state_encoder(s, hidden) if hidden else self.state_encoder(s)
                next_state = next_state.squeeze()

                if np.random.random() > 0.95 and state is not None:
                    batch = [state, action, reward, next_state]
                    buffer.append(batch)

        loss = learning.ddpg_update(batch, self.params, self.optimizer, self.device, self.debugger,
                                    step=self.step, learn=False)
        buffer.flush()
        return loss

    def train(self):
        max_buf_size = 100000
        buffer = misc.ReplayBuffer(max_buf_size)

        for batch in self.dataloader['train']:
            batch = [i.to(self.device) for i in batch]
            items, ratings, sizes = batch
            hidden = None
            state = None
            for t in range(int(sizes.min().item()) - 1):
                action = items[:, t]
                reward = ratings[:, t].unsqueeze(-1)
                s = torch.cat([action, reward], 1).unsqueeze(0)
                next_state, hidden = self.state_encoder(s, hidden) if hidden else self.state_encoder(s)
                next_state = next_state.squeeze()

                if np.random.random() > 0.95 and state is not None:
                    batch = [state, action, reward, next_state]
                    buffer.append(batch)

                if buffer.len() >= max_buf_size:
                    loss = learning.ddpg_update(batch, self.params, self.optimizer, self.device, self.debugger,
                                                step=self.step, learn=False)
                    self.debugger.log_losses(loss)
                    self.step += 1
                    self.debugger.log_step(self.step)
                    buffer.flush()

                state = next_state

    def get_parameters(self):
        if self.state_encoder:
            return list(self.state_encoder.parameters()),\
                   list(self.policy_net.parameters()), \
                   list(self.value_net.parameters())

        return list(self.policy_net.parameters()), list(self.value_net.parameters())

    """
    You need to manually register optimizers you like
    ddpg = DDPG(...)
    state, policy, value = ddpg.get_parameters()
    value_optimizer = optim.RAdam(value, lr=params['value_lr'], weight_decay=1e-2)
    policy_optimizer = optim.RAdam(state + policy, lr=params['policy_lr'], weight_decay=1e-2)
    optimizer = {
        'policy_optimizer': policy_optimizer,
        'value_optimizer':  value_optimizer
    }
    ddpg.register_optimizers(optimizer)
    """

    def register_optimizers(self, optimizer):
        self.optimizer = optimizer

    def init_state_encoder(self, state_encoder, env):
        if state_encoder == 'none':
            self.state_encoder = False
        elif state_encoder == 'linear':
            self.state_encoder = nn.Sequential(nn.Linear(env['original_state_dim'],
                                                         env['state_dim']), nn.Tanh()).to(self.device)
        elif state_encoder == 'lstm':
            self.state_encoder = nn.LSTM(env['original_state_dim'], env['state_dim'],
                                         batch_first=True).to(self.device)

        else:
            raise NotImplementedError('You should specify the encoder type')

    def update(self, batch):
        if self.optimizer is None:
            raise ReferenceError('Optimizers are not provided! You need to register them!')
        learning.ddpg_update(batch, self.params, self.optimizer, self.device, self.debugger)
