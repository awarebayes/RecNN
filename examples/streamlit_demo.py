import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import copy
from tqdm.auto import tqdm

import torch
from scipy.spatial import distance

# == recnn ==
import sys
sys.path.append("../")
import recnn
tqdm.pandas()

import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server

# constants

RATINGSPATH = '../data/ml-20m/ratings.csv'
EMBEDPATH =  '../data/embeddings/ml20_pca128.pkl'
METAPATH = '../data/parsed/omdb.json'
MODELSPATH = '../models/'

# huge thanks to https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92 !
class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get(**kwargs):
        ctx = ReportThread.get_report_ctx()

        session = None
        session_infos = Server.get_current()._session_infos.values()

        for session_info in session_infos:
            if session_info.session._main_dg == ctx.main_dg:
                session = session_info.session

        if session is None:
            raise RuntimeError(
                "Oh noes. Couldn't get your Streamlit Session object"
                'Are you doing something fancy with threads?')

        if not getattr(session, '_custom_session_state', None):
            session._custom_session_state = SessionState(**kwargs)

        return session._custom_session_state

@st.cache
def set_up_env():
    env = recnn.data.FrameEnv(EMBEDPATH, RATINGSPATH, 10, 1)

    batch_wrapper = env.prepare_batch_wrapper
    del env.prepare_batch_wrapper, env.train_dataloader.collate_fn, env.test_dataloader.collate_fn

    env_serialized = pickle.dumps(env)
    return env_serialized, batch_wrapper


def get_env():
    env, batch_wrapper = set_up_env()
    env = pickle.loads(env)
    env.prepare_batch_wrapper = batch_wrapper
    env.train_dataloader.collate_fn = batch_wrapper
    env.test_dataloader.collate_fn = batch_wrapper

    return env

@st.cache
def load_omdb_meta():
    return json.load(open(METAPATH))

def load_models():
    state = SessionState.get(env=None, meta=None, models=None, device=None)
    ddpg = recnn.nn.models.Actor(1290, 128, 256).to(state.device)
    td3 = recnn.nn.models.Actor(1290, 128, 256).to(state.device)
    ddpg.load_state_dict(torch.load( MODELSPATH + 'ddpg_policy.pt'))
    td3.load_state_dict(torch.load(MODELSPATH + 'td3_policy.pt'))
    return {'ddpg': ddpg, 'td3': td3}

def rank(gen_action, metric):
    scores = []
    state = SessionState.get(env=None, meta=None, models=None, device=None)
    env = state.env
    meta = state.meta
    for i in env.movie_embeddings_key_dict.keys():
        if i == 0 or i == '0':
            continue
        scores.append([i, metric(env.movie_embeddings_key_dict[i], gen_action)])
    scores = list(sorted(scores, key = lambda x: x[1]))
    scores = scores[:10]
    ids = [i[0] for i in scores]
    for i in range(10):
        #scores[i].extend([meta[str(scores[i][0])]['omdb'][key]  for key in ['Title',
        #                        'Genre', 'Language', 'Released', 'imdbRating']])
        scores[i].extend([meta[str(scores[i][0])]['omdb'][key]  for key in ['Title',
                                'Genre', 'imdbRating']])
        # scores[i][3] = ' '.join([genres_dict[i] for i in scores[i][3]])

    # indexes = ['id', 'score', 'Title', 'Genre', 'Language', 'Released', 'imdbRating']
    indexes = ['id', 'score', 'Title', 'Genre', 'imdbRating']
    table_dict = dict([(key,[i[idx] for i in scores]) for idx, key in enumerate(indexes)])
    table = pd.DataFrame(table_dict)
    return table

def main():

    storage = SessionState.get(env=None, meta=None, models=None, device=None)

    st.title("Welcome to recnn's official demo!")

    # ====== 1. Let's get your data (down)loaded") ======

    st.header("1. Let's get your data (down)loaded")

    if st.checkbox('Use cuda', True):
        storage.device = torch.device('cuda')
    else:
        storage.device = torch.device('cpu')

    st.markdown(
        """
        ### Downloads
        - [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
        - [My Movie Embeddings](https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL)
        """
    )

    st.subheader('Loads')
    global RATINGSPATH, EMBEDPATH, METAPATH, MODELSPATH
    RATINGSPATH = st.text_input('Enter the path to ratings.csv from ML20:', '../data/ml-20m/ratings.csv')
    EMBEDPATH = st.text_input('Enter the path to my embeddings:', '../data/embeddings/ml20_pca128.pkl')
    METAPATH = st.text_input('Enter the path OMDB data:', '../data/parsed/omdb.json')

    st.info("Unfortunately there is no progress verbose in streamlit. Look in your console!")
    if st.button("Start loading!"):
        storage.env = get_env()
        storage.meta = load_omdb_meta()
        st.success('Data is loaded!')
        flag = 1

    # ====== 3. Load the models ======

    st.header("3. Load the models")

    st.markdown("""
        | Algorithm                             | Paper                            | Code                       |
        |---------------------------------------|----------------------------------|----------------------------|
        | Deep Q Learning (PoC)                 | https://arxiv.org/abs/1312.5602  | examples/0. Embeddings/ 1.DQN |
        | Deep Deterministic Policy Gradients   | https://arxiv.org/abs/1509.02971 | examples/1.Vanilla RL/DDPG |
        | Twin Delayed DDPG (TD3)               | https://arxiv.org/abs/1802.09477 | examples/1.Vanilla RL/TD3  |
        | Soft Actor-Critic                     | https://arxiv.org/abs/1801.01290 | examples/1.Vanilla RL/SAC  |
        | Batch Constrained Q-Learning          | https://arxiv.org/abs/1812.02900 | examples/99.To be released/BCQ |
        | REINFORCE Top-K Off-Policy Correction | https://arxiv.org/abs/1812.02353 | examples/2. REINFORCE TopK |
    """)

    st.subheader('')

    MODELSPATH = st.text_input('Enter the path to pre-trained models folder', '../models/')

    if st.button("Start loading models!"):
        storage.models = load_models()
        st.success('Models are loaded!')

    # ====== 4. Sample ======

    st.header("4. Sample a Batch")
    st.subheader("Let's sample a batch")
    if st.button("Sample batch"):
        storage.test_batch = next(iter(storage.env.test_dataloader))
        state, action, reward, next_state, done = recnn.data.get_base_batch(storage.test_batch)

        st.subheader('State')
        st.write(state)
        st.subheader('Action')
        st.write(action)
        st.subheader('Reward')
        st.write(reward.squeeze())

    # ====== 5. Recommend ======

    st.header('5. Recommend')

    algorithm = st.selectbox('Choose an algorithm',  ('ddpg', 'td3'))
    metric = st.selectbox('Choose a metric', ('euclidean', 'cosine', 'correlation',
                                              'canberra', 'minkowski', 'chebyshev',
                                              'braycurtis', 'cityblock', ))
    dist = {'euclidean': distance.euclidean, 'cosine': distance.cosine,
            'correlation': distance.correlation, 'canberra': distance.canberra,
            'minkowski': distance.minkowski, 'chebyshev': distance.chebyshev,
            'braycurtis': distance.braycurtis, 'cityblock': distance.cityblock}
    if st.button('Rank'):
        state, _, _, _, _ = recnn.data.get_base_batch(storage.test_batch)
        rand_id = np.random.randint(0, state.size(0), 1)[0]
        st.markdown('**Recommendations for state with index {} (random)**'.format(rand_id))
        action = storage.models[algorithm].forward(state)
        # pick random action
        action = action[rand_id].detach().cpu().numpy()
        st.write(rank(action, dist[metric]))


if __name__ == "__main__":
    main()