import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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
        :param action: nothing should be provided here.
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


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size, init_w=0):
        super(DiscreteActor, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_dim)

        self.saved_log_probs = []
        self.rewards = []
        self.correction = []
        self.lambda_k = []

        # What's action source? See this issue: https://github.com/awarebayes/RecNN/issues/7
        # by default {pi: pi, beta: beta}
        # you can change it to be like {pi: beta, beta: beta} as miracle24 suggested

        self.action_source = {"pi": "pi", "beta": "beta"}
        self.select_action = self._select_action

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)

    def gc(self):
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.correction[:]
        del self.lambda_k[:]

    def _select_action(self, state, **kwargs):

        # for reinforce without correction only pi_probs is available.
        # the action source is ignored, since there is no beta

        pi_probs = self.forward(state)
        pi_categorical = Categorical(pi_probs)
        pi_action = pi_categorical.sample()
        self.saved_log_probs.append(pi_categorical.log_prob(pi_action))
        return pi_probs

    def pi_beta_sample(self, state, beta, action, **kwargs):
        # 1. obtain probabilities
        # note: detach is to block gradient
        beta_probs = beta(state.detach(), action=action)
        pi_probs = self.forward(state)

        # 2. probabilities -> categorical distribution.
        beta_categorical = Categorical(beta_probs)
        pi_categorical = Categorical(pi_probs)

        # 3. sample the actions
        # See this issue: https://github.com/awarebayes/RecNN/issues/7
        # usually it works like:
        # pi_action = pi_categorical.sample(); beta_action = beta_categorical.sample();
        # but changing the action_source to {pi: beta, beta: beta} can be configured to be:
        # pi_action = beta_categorical.sample(); beta_action = beta_categorical.sample();
        available_actions = {
            "pi": pi_categorical.sample(),
            "beta": beta_categorical.sample(),
        }
        pi_action = available_actions[self.action_source["pi"]]
        beta_action = available_actions[self.action_source["beta"]]

        # 4. calculate stuff we need
        pi_log_prob = pi_categorical.log_prob(pi_action)
        beta_log_prob = beta_categorical.log_prob(beta_action)

        return pi_log_prob, beta_log_prob, pi_probs

    def _select_action_with_correction(
        self, state, beta, action, writer, step, **kwargs
    ):
        pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)

        # calculate correction
        corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)

        writer.add_histogram("correction", corr, step)
        writer.add_histogram("pi_log_prob", pi_log_prob, step)
        writer.add_histogram("beta_log_prob", beta_log_prob, step)

        self.correction.append(corr)
        self.saved_log_probs.append(pi_log_prob)

        return pi_probs

    def _select_action_with_TopK_correction(
        self, state, beta, action, K, writer, step, **kwargs
    ):
        pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)

        # calculate correction
        corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)

        # calculate top K correction
        l_k = K * (1 - torch.exp(pi_log_prob)) ** (K - 1)

        writer.add_histogram("correction", corr, step)
        writer.add_histogram("l_k", l_k, step)
        writer.add_histogram("pi_log_prob", pi_log_prob, step)
        writer.add_histogram("beta_log_prob", beta_log_prob, step)

        self.correction.append(corr)
        self.lambda_k.append(l_k)
        self.saved_log_probs.append(pi_log_prob)

        return pi_probs


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
        z = mean + std * self.normal.sample(std.size()).to(
            next(self.parameters()).device
        )

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
