

# https://spinningup.openai.com/en/latest/algorithms/sac.html
def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = None
        self.idx = 0
        self.size = buffer_size
        self.flush()

    def flush(self):
        # state, action, reward, next_state
        self.buffer = [torch.zeros(self.size, 256),
                       torch.zeros(self.size, 128),
                       torch.zeros(self.size, 1),
                       torch.zeros(self.size, 256)]
        self.idx = 0

    def append(self, batch):
        state, action, reward, next_state = batch
        lower = self.idx
        upper = state.size(0) + lower
        self.buffer[0][lower:upper] = state
        self.buffer[1][lower:upper] = action
        self.buffer[2][lower:upper] = reward
        self.buffer[3][lower:upper] = next_state
        self.idx += upper

    def get(self):
        return self.buffer

    def len(self):
        return self.idx

