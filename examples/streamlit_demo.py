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

st.title("Welcome to recnn's official demo!")

# ====== 1. Let's get your data (down)loaded") ======

st.header("1. Let's get your data (down)loaded")

st.markdown(
    """
    ### Downloads
    - [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
    - [My Movie Embeddings](https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL)
    """
)

st.subheader('Loads')
RATINGSPATH = st.text_input('Enter the path to ratings.csv from ML20:', '../data/ml-20m/ratings.csv')
EMBEDPATH = st.text_input('Enter the path to my embeddings:', '../data/embeddings/ml20_pca128.pkl')
METAPATH = st.text_input('Enter the path OMDB data:', '../data/parsed/omdb.json')
MODELSPATH = st.text_input('Enter the path to pre-trained models folder', '../models/')

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

def main():
    st.info("Unfortunately there is no progress verbose in streamlit. Look in your console!")
    state = SessionState.get(env=None, meta=None, models=None, device=None)
    if st.button("Start loading!"):
        state.env = get_env()
        state.meta = load_omdb_meta()
        st.success('Data is loaded!')
        flag = 1

    # ====== 2. Configure ======

    st.header("2. Configure")

    if st.checkbox('Use cuda', True):
        state.device = torch.device('cuda')
    else:
        state.device = torch.device('cpu')

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

    if st.button("Start loading models!"):
        state.models = load_models()
        st.success('Models are loaded!')

    # ====== 4. Sample ======

    st.header("4. Sample")
    st.subheader("Let's sample a batch")
    if st.button("Sample batch"):
        state.test_batch = next(iter(state.env.test_dataloader))
        state, action, reward, next_state, done = recnn.data.get_base_batch(state.test_batch)

        st.subheader('State')
        st.write(state)
        st.subheader('Action')
        st.write(action)

if __name__ == "__main__":
    main()