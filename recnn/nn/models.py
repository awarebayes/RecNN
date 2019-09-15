import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetector(nn.Module):

    """
    Anomaly detector used for debugging. Basically an auto encoder.
    P.S. You need to use different weights for different embeddings.
    """

    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.ae = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        """"""
        return self.ae(x)

    def rec_error(self, x):
        error = torch.sum((x - self.ae(x)) ** 2, 1)
        if x.size(1) != 1:
            return error.detach()
        return error.item()


class Actor(nn.Module):

    """
    Vanilla actor. Takes state as an argument, returns action.
    """

    def __init__(self, input_dim, action_dim, hidden_size, init_w=2e-1):
        super(Actor, self).__init__()
        
        self.drop_layer = nn.Dropout(p=0.5)
        
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, tanh=False):
        """
        :param state: state
        :param tanh: whether to use tahn as action activation
        :return: action
        """
        action = F.relu(self.linear1(state))
        action = self.drop_layer(action)
        action = F.relu(self.linear2(action))
        action = self.drop_layer(action)
        action = self.linear3(action)
        if tanh:
            action = F.tanh(action)
        return action


class Critic(nn.Module):

    """
    Vanilla critic. Takes state and action as an argument, returns value.
    """

    def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-5):
        super(Critic, self).__init__()
        
        self.drop_layer = nn.Dropout(p=0.5)
        
        self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        """"""
        value = torch.cat([state, action], 1)
        value = F.relu(self.linear1(value))
        value = self.drop_layer(value)
        value = F.relu(self.linear2(value))
        value = self.drop_layer(value)
        value = self.linear3(value)
        return value


class bcqPerturbator(nn.Module):

    """
    Batch constrained perturbative actor. Takes action as an argument, adjusts it.
    """

    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-1):
        super(bcqPerturbator, self).__init__()
        
        self.drop_layer = nn.Dropout(p=0.5)
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        """"""
        a = torch.cat([state, action], 1)
        a = F.relu(self.linear1(a))
        a = self.drop_layer(a)
        a = F.relu(self.linear2(a))
        a = self.drop_layer(a)
        a = self.linear3(a) 
        return a + action


class bcqGenerator(nn.Module):

    """
    Batch constrained generator. Basically VAE
    """

    def __init__(self, state_dim, action_dim, latent_dim):
        super(bcqGenerator, self).__init__()
        # encoder
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)
        
        # decoder
        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)
        
        self.latent_dim = latent_dim
        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, state, action):
        """"""
        # z is encoded state + action
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * self.normal.sample(std.size()).to(next(self.parameters()).device)

        # u is decoded action
        u = self.decode(state, z)

        return u, mean, std
    
    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = self.normal.sample([state.size(0), self.latent_dim])
            z = z.clamp(-0.5, 0.5).to(next(self.parameters()).device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.d3(a)