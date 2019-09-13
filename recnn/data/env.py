from . import utils
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm.auto import tqdm


class UserDataset(Dataset):

    """
    torch.DataSet
    users:arg - list of user id's
    user_dict:arg - dict {user_id: {
                                    'items': [item_id (np.ndarray)],
                                    'ratings': [ratings (np.ndarray)]
                                    } }
    """

    def __init__(self, users, user_dict):
        self.users = users
        self.user_dict = user_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        idx = self.users[idx]
        group = self.user_dict[idx]
        items = group['items'][:]
        rates = group['ratings'][:]
        size = items.shape[0]
        return {'items': items, 'rates': rates, 'sizes': size}


class Env:
    def __init__(self, embeddings, ratings, train_ratio=0.95, min_seq_size=10, data_cols={}):
        self.movie_embeddings_key_dict = pickle.load(open(embeddings, 'rb'))
        movies_embeddings_tensor, key_to_id, id_to_key = utils.make_items_tensor(self.movie_embeddings_key_dict)
        self.embeddings = movies_embeddings_tensor
        self.key_to_id = key_to_id
        self.id_to_key = id_to_key
        self.ratings = pd.read_csv(ratings)
        user_dict, users = utils.prepare_dataset(self.ratings, self.key_to_id, min_seq_size, **data_cols)
        self.user_dict = user_dict
        self.users = users
        train_ratio = int(len(users) * train_ratio)
        self.train_ratio = train_ratio

        self.test_users = users[train_ratio:]
        self.train_users = users[:train_ratio]
        self.train_users = utils.sort_users_itemwise(self.user_dict, self.train_users)[2:]
        self.test_users = utils.sort_users_itemwise(self.user_dict, self.test_users)
        self.train_user_dataset = UserDataset(self.train_users, self.user_dict)
        self.test_user_dataset = UserDataset(self.test_users, self.user_dict)


class FrameEnv(Env):
    def __init__(self, embeddings, ratings, frame_size, batch_size):
        super(FrameEnv, self).__init__(embeddings, ratings, min_seq_size=frame_size+1)

        def prepare_batch_wrapper(x):
            batch = utils.prepare_batch_static_size(x, self.embeddings, frame_size=frame_size)
            return batch

        self.train_dataloader = DataLoader(self.train_user_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=1, collate_fn=prepare_batch_wrapper)

        self.test_dataloader = DataLoader(self.test_user_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=1, collate_fn=prepare_batch_wrapper)

    def train_batch(self):
        return next(iter(self.train_dataloader))

    def test_batch(self):
        return next(iter(self.test_dataloader))


class SeqEnv(Env):

    def __init__(self, embeddings, ratings, batch_size, state_encoder, device, max_buf_size=1000):
        super(SeqEnv, self).__init__(embeddings, ratings, min_seq_size=10)

        def prepare_batch_wrapper(batch):
            batch = utils.padder(batch)
            batch = utils.prepare_batch_dynamic_size(batch, self.embeddings)
            return batch

        self.device = device
        self.state_encoder = state_encoder
        self.max_buf_size = max_buf_size
        self.train_dataloader = DataLoader(self.train_user_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=1, collate_fn=prepare_batch_wrapper)
        self.test_dataloader = DataLoader(self.test_user_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=1, collate_fn=prepare_batch_wrapper)

        self.buffer_layout = [torch.zeros(self.max_buf_size, 256),
                              torch.zeros(self.max_buf_size, 128),
                              torch.zeros(self.max_buf_size, 1),
                              torch.zeros(self.max_buf_size, 256)]

        self.train_buffer = utils.ReplayBuffer(self.max_buf_size)
        self.test_buffer  = utils.ReplayBuffer(self.max_buf_size)

    def train_batch(self):
        while 1:
            for batch in tqdm(self.train_dataloader):
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
                        self.train_buffer.append(batch)

                    if self.train_buffer.len() >= self.max_buf_size:
                        g = self.train_buffer.get()
                        self.train_buffer.flush()
                        yield g

                    state = next_state

    def test_batch(self):
        while 1:
            for batch in tqdm(self.test_dataloader):
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
                        self.test_buffer.append(batch)

                    if self.test_buffer.len() >= self.max_buf_size:
                        g = self.test_buffer.get()
                        self.test_buffer.flush()
                        yield g
                        del g

                    state = next_state
