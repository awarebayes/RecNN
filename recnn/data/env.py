from . import utils
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

"""
.. module:: env
   :synopsis: Main abstraction of the library for datasets is called environment, similar to how other reinforcement
    learning libraries name it. This interface is created to provide SARSA like input for your RL Models. When you are 
    working with recommendation env, you have two choices: using static length inputs (say 10 items) or dynamic length 
    time series with sequential encoders (many to one rnn). Static length is provided via FrameEnv, and dynamic length
    along with sequential state representation encoder is implemented in SeqEnv. Let’s take a look at FrameEnv first:


.. moduleauthor:: Mike Watts <awarebayes@gmail.com>


"""


class UserDataset(Dataset):

    """
    Low Level API: dataset class user: [items, ratings], Instance of torch.DataSet
    """

    def __init__(self, users, user_dict):
        """

        :param users: integer list of user_id. Useful for train/test splitting
        :type users: list<int>.
        :param user_dict: dictionary of users with user_id as key and [items, ratings] as value
        :type user_dict: (dict{ user_id<int>: dict{'items': list<int>, 'ratings': list<int>} }).

        """

        self.users = users
        self.user_dict = user_dict

    def __len__(self):
        """
        useful for tqdm, consists of a single line:
        return len(self.users)
        """
        return len(self.users)

    def __getitem__(self, idx):
        """
        getitem is a function where non linear user_id maps to a linear index. For instance in the ml20m dataset,
        there are big gaps between neighbouring user_id. getitem removes these gaps, optimizing the speed.

        :param idx: index drawn from range(0, len(self.users)). User id can be not linear, idx is.
        :type idx: int

        :returns:  dict{'items': list<int>, rates:list<int>, sizes: int}
        """
        idx = self.users[idx]
        group = self.user_dict[idx]
        items = group['items'][:]
        rates = group['ratings'][:]
        size = items.shape[0]
        return {'items': items, 'rates': rates, 'sizes': size, 'users': idx}


class Env:

    """
    Env abstract class
    """

    def __init__(self, embeddings, ratings, test_size=0.05, min_seq_size=10, prepare_dataset=utils.prepare_dataset):

        """
        .. note::
            embeddings need to be provided in {movie_id: torch.tensor} format!

        :param embeddings: path to where item embeddings are stored.
        :type embeddings: str
        :param ratings: path to the dataset that is similar to the ml20m
        :type ratings: str
        :param test_size: ratio of users to use in testing. Rest will be used for training/validation
        :type test_size: int
        :param min_seq_size: filter users: len(user.items) > min seq size
        :type min_seq_size: int
        :param prepare_dataset: function you provide. should yield user_dict, users
        :type prepare_dataset: function
        """

        self.prepare_dataset = prepare_dataset
        self.movie_embeddings_key_dict = pickle.load(open(embeddings, 'rb'))
        movies_embeddings_tensor, key_to_id, id_to_key = utils.make_items_tensor(self.movie_embeddings_key_dict)
        self.embeddings = movies_embeddings_tensor
        self.key_to_id = key_to_id
        self.id_to_key = id_to_key
        self.ratings = pd.read_csv(ratings)
        user_dict, users = self.prepare_dataset(self.ratings, self.key_to_id, min_seq_size)
        self.user_dict = user_dict
        self.users = users  # filtered keys of user_dict

        self.train_users, self.test_users = train_test_split(users, test_size=test_size)
        self.train_users = utils.sort_users_itemwise(self.user_dict, self.train_users)[2:]
        self.test_users = utils.sort_users_itemwise(self.user_dict, self.test_users)
        self.train_user_dataset = UserDataset(self.train_users, self.user_dict)
        self.test_user_dataset = UserDataset(self.test_users, self.user_dict)


class FrameEnv(Env):
    """
    Static length user environment.
    """
    def __init__(self, embeddings, ratings, frame_size=10, batch_size=25, *args, **kwargs):

        """
        :param embeddings: path to where item embeddings are stored.
        :type embeddings: str
        :param ratings: path to the dataset that is similar to the ml20m
        :type ratings: str
        :param frame_size: len of a static sequence, frame
        :type frame_size: int

        p.s. you can also provide **pandas_conf in the arguments.

        It is useful if you dataset columns are different from ml20::

            pandas_conf = {user_id='userId', rating='rating', item='movieId', timestamp='timestamp'}
            env = FrameEnv(embed_dir, rating_dir, **pandas_conf)

        """

        super(FrameEnv, self).__init__(embeddings, ratings, min_seq_size=frame_size+1, *args, **kwargs)

        def prepare_batch_wrapper(x):
            batch = utils.prepare_batch_static_size(x, self.embeddings, frame_size=frame_size)
            return batch

        self.train_dataloader = DataLoader(self.train_user_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=1, collate_fn=prepare_batch_wrapper)

        self.test_dataloader = DataLoader(self.test_user_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=1, collate_fn=prepare_batch_wrapper)

    def train_batch(self):
        """ Get batch for training """
        return next(iter(self.train_dataloader))

    def test_batch(self):
        """ Get batch for testing """
        return next(iter(self.test_dataloader))


class SeqEnv(Env):

    """
    Dynamic length user environment.
    Due to some complications, this module is implemented quiet differently from FrameEnv.
    First of all, it relies on the replay buffer. Train/Test batch is a generator.
    In batch generator, I iterate through the batch, and choose target action with certain probability.
    Hence, ~95% is state that is encoded with state encoder and ~5% are actions.
    If you have a better solution, your contribution is welcome
    """

    def __init__(self, embeddings, ratings,  state_encoder, batch_size=25, device=torch.device('cuda'),
                                                                    layout=None, max_buf_size=1000):

        """
        :param embeddings: path to where item embeddings are stored.
        :type embeddings: str
        :param ratings: path to the dataset that is similar to the ml20m
        :type ratings: str
        :param state_encoder: state encoder of your choice
        :type state_encoder: nn.Module
        :param device: device of your choice
        :type device: torch.device
        :param max_buf_size: maximum size of a replay buffer
        :type max_buf_size: int
        :param layout: how sizes in batch should look like
        :type layout: list<torch.Size>
        """

        super(SeqEnv, self).__init__(embeddings, ratings, min_seq_size=10)

        if layout is None:
            # default ml20m layout for my (action=128) embeddings, (state=256)
            layout = [torch.Size([max_buf_size, 256]),
                      torch.Size([max_buf_size, 128]),
                      torch.Size([max_buf_size, 1]),
                      torch.Size([max_buf_size, 256])]

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

        self.buffer_layout = layout

        self.train_buffer = utils.ReplayBuffer(self.max_buf_size, layout=self.buffer_layout)
        self.test_buffer = utils.ReplayBuffer(self.max_buf_size, layout=self.buffer_layout)

    def train_batch(self):
        while 1:
            for batch in tqdm(self.train_dataloader):
                items, ratings, sizes, users = utils.get_irsu(batch)
                items, ratings, sizes = [i.to(self.device) for i in [items, ratings, sizes]]
                hidden = None
                state = None
                self.train_buffer.meta.update({'sizes': sizes, 'users': users})
                for t in range(int(sizes.min().item()) - 1):
                    action = items[:, t]
                    reward = ratings[:, t].unsqueeze(-1)
                    s = torch.cat([action, reward], 1).unsqueeze(0)
                    next_state, hidden = self.state_encoder(s, hidden) if hidden else self.state_encoder(s)
                    next_state = next_state.squeeze()

                    if np.random.random() > 0.95 and state is not None:
                        batch = {'state': state, 'action': action, 'reward': reward,
                                 'next_state': next_state, 'step': t}
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
