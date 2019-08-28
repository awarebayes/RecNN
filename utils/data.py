from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class UserDataset(Dataset):

    def __init__(self, users, user_dict):
        self.users = users
        self.user_dict = user_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        idx = self.users[idx]
        group = self.user_dict[idx]
        movies = group['movies'][:]
        rates = group['ratings'][:]
        size = movies.shape[0]
        return {'movies': movies, 'rates': rates, 'sizes': size}
    
    
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# name speaks for itself
def prepare_batch(batch, movie_ref_tensor, frame_size=10):
    
    movies_t, ratings_t, sizes_t = [], [], []
    for i in range(len(batch)):
        movies_t.append(batch[i]['movies'])
        ratings_t.append(batch[i]['rates'])
        sizes_t.append(batch[i]['sizes'])  

    movies_t = np.concatenate([rolling_window(i, frame_size+1) for i in movies_t], 0)
    ratings_t = np.concatenate([rolling_window(i, frame_size+1) for i in ratings_t], 0)
    
    movies_t = torch.tensor(movies_t)
    ratings_t = torch.tensor(ratings_t).float()
    sizes_t = torch.tensor(sizes_t)
    
    batch_size = ratings_t.size(0)
    
    movies_tensor = movie_ref_tensor[movies_t.long()]
    
    movies = movies_tensor[:, :-1, :].view(batch_size, -1)
    next_movies = movies_tensor[:, 1:, :].view(batch_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]
    
    state = torch.cat([movies, ratings], 1)
    next_state = torch.cat([next_movies, next_ratings], 1)
    action = movies_tensor[:, -1, :]
    reward = ratings_t[:, -1]
    
    done = torch.zeros(batch_size)
    done[torch.cumsum(sizes_t-frame_size, dim=0)-1] = 1

    return state, action, reward, next_state, done


def make_movie_tensor(movie_ref):
    
    movie_ref[0] = torch.zeros(128)
    keys = list(sorted(movie_ref.keys()))
    key_to_id = dict(zip(keys, range(len(keys))))
    id_to_key = dict(zip(range(len(keys)), keys))

    movie_ref_dict = {}
    for k in movie_ref.keys():
        movie_ref_dict[key_to_id[k]] = movie_ref[k]
    movie_ref_tensor = torch.stack([movie_ref_dict[i] for i in range(len(movie_ref_dict))])
    return movie_ref_tensor, key_to_id, id_to_key

def prepare_tensor(ratings, key_to_id, frame_size):
    ratings["rating"] = ratings["rating"].progress_apply(lambda i: 2 * (i - 2.5))
    ratings["movieId"] = ratings["movieId"].progress_apply(key_to_id.get)
    users = ratings[["userId","movieId"]].groupby(["userId"]).size()
    users = users[users >= frame_size + 1]
    users = users.index
    ratings = ratings.sort_values(by='timestamp').set_index("userId").drop("timestamp", axis=1).groupby('userId')

    # Groupby user
    user_dict = {}
    def app(x):
        userid = x.index[0]
        user_dict[int(userid)] = {}
        user_dict[int(userid)]['movies'] = x['movieId'].values
        user_dict[int(userid)]['ratings'] = x['rating'].values
    ratings.progress_apply(app)
    return user_dict, users