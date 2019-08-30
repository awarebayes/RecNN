from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


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
    

# helper function similar to pandas.Series.rolling
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def batch_no_embeddings(batch, frame_size):
    item_t, ratings_t, sizes_t = batch
    b_size = ratings_t.size(0)
    items = item_t[:, :-1]
    next_items = item_t[:, 1:]
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]
    action = item_t[:, -1]
    reward = ratings_t[:, -1]
    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1
    return items, next_items, ratings, next_ratings, action, reward, done


def batch_tensor_embeddings(batch, item_embeddings_tensor, frame_size):
    item_t, ratings_t, sizes_t = batch
    items_tensor = item_embeddings_tensor[item_t.long()]
    b_size = ratings_t.size(0)

    items = items_tensor[:, :-1, :].view(b_size, -1)
    next_items = items_tensor[:, 1:, :].view(b_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]

    state = torch.cat([items, ratings], 1)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_tensor[:, -1, :]
    reward = ratings_t[:, -1]

    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1

    return state, action, reward, next_state, done


# Main function that is used as torch.DataLoader->collate_fn
# CollateFn docs:
# https://pytorch.org/docs/stable/data.html#working-with-collate-fn
def prepare_batch_static_size(batch, item_embeddings_tensor=False, frame_size=10):
    item_t, ratings_t, sizes_t = [], [], []
    for i in range(len(batch)):
        item_t.append(batch[i]['items'])
        ratings_t.append(batch[i]['rates'])
        sizes_t.append(batch[i]['sizes'])

    item_t = np.concatenate([rolling_window(i, frame_size + 1) for i in item_t], 0)
    ratings_t = np.concatenate([rolling_window(i, frame_size + 1) for i in ratings_t], 0)

    item_t = torch.tensor(item_t)
    ratings_t = torch.tensor(ratings_t).float()
    sizes_t = torch.tensor(sizes_t)

    batch_size = ratings_t.size(0)

    if type(item_embeddings_tensor) == bool:
        return batch_no_embeddings([item_t, ratings_t, sizes_t], frame_size)
    elif type(item_embeddings_tensor) == torch.Tensor:
        return batch_tensor_embeddings([item_t, ratings_t, sizes_t], item_embeddings_tensor, frame_size)


# Usually in data sets there item index is inconsistent (if you plot it doesn't look like a line)
# This function makes the index linear, allows for better compression of the data
# And also makes use of tensor[tensor] semantics

# items_embeddings_key_dict:arg - item embeddings by key
# include_zero:arg - whether to include items_embeddings_id_dict[0] = [0, 0, 0, ..., 0] (128)
# sometimes needed for rnn padding, by default True
# returns:
# items_embeddings_tensor - items_embeddings_dict compressed into tensor
# key_to_id - dict key -> index
# id_to_key - dict index -> key


def make_items_tensor(items_embeddings_key_dict, include_zero=True):
    if include_zero:
        items_embeddings_key_dict[0] = torch.zeros(128)
    keys = list(sorted(items_embeddings_key_dict.keys()))
    key_to_id = dict(zip(keys, range(len(keys))))
    id_to_key = dict(zip(range(len(keys)), keys))

    items_embeddings_id_dict = {}
    for k in items_embeddings_key_dict.keys():
        items_embeddings_id_dict[key_to_id[k]] = items_embeddings_key_dict[k]
    items_embeddings_tensor = torch.stack([items_embeddings_id_dict[i] for i in range(len(items_embeddings_id_dict))])
    return items_embeddings_tensor, key_to_id, id_to_key


"""
    Main function used for dataset transformation
    Basically works like pandas.groupby user 
    Arguments:
        df:arg - ml20 like dataset
        key_to_id:arg - 
        frame_size:arg - only used for static size batches, number of items to take
        user_id:arg string name of 'user id' pandas column
        rating:arg string name of 'rating' pandas column
        item:arg string name of 'item id' pandas column
    Returns:
        user_dict:arg - dict {user_id: {
                                    'items': [item_id (np.ndarray)],
                                    'ratings': [ratings (np.ndarray)]
                                } }
        users: list of valid users (n_items > frame_size)
"""


def prepare_dataset(df, key_to_id, frame_size, user_id='userId', rating='rating', item='movieId'):
    df[rating] = df[rating].progress_apply(lambda i: 2 * (i - 2.5))
    df[item] = df[item].progress_apply(key_to_id.get)
    users = df[[user_id, item]].groupby([user_id]).size()
    users = users[users > frame_size]
    users = users.index
    ratings = df.sort_values(by='timestamp').set_index(user_id).drop("timestamp", axis=1).groupby(user_id)

    # Groupby user
    user_dict = {}

    def app(x):
        userid = x.index[0]
        user_dict[int(userid)] = {}
        user_dict[int(userid)]['items'] = x[item].values
        user_dict[int(userid)]['ratings'] = x[rating].values

    ratings.progress_apply(app)
    return user_dict, users
