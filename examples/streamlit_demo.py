import streamlit as st
import matplotlib.pyplot as plt
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

# disable it if you get an error
from jupyterthemes import jtplot
jtplot.style(theme='grade3')

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
def get_mekd():
    return get_env().movie_embeddings_key_dict


@st.cache
def get_batch(device):
    env = get_env()
    test_batch = next(iter(env.test_dataloader))
    return recnn.data.get_base_batch(test_batch, device=device)


@st.cache
def load_omdb_meta():
    return json.load(open(METAPATH))


def load_models(device):
    ddpg = recnn.nn.models.Actor(1290, 128, 256).to(device)
    td3 = recnn.nn.models.Actor(1290, 128, 256).to(device)
    ddpg.load_state_dict(torch.load( MODELSPATH + 'ddpg_policy.pt', map_location=device))
    td3.load_state_dict(torch.load(MODELSPATH + 'td3_policy.pt', map_location=device))
    return {'ddpg': ddpg, 'td3': td3}


def rank(gen_action, metric, k):
    scores = []
    movie_embeddings_key_dict = get_mekd()
    meta = load_omdb_meta()

    for i in movie_embeddings_key_dict.keys():
        if i == 0 or i == '0':
            continue
        scores.append([i, metric(movie_embeddings_key_dict[i], gen_action)])
    scores = list(sorted(scores, key = lambda x: x[1]))
    scores = scores[:k]
    ids = [i[0] for i in scores]
    for i in range(k):
        #scores[i].extend([meta[str(scores[i][0])]['omdb'][key]  for key in ['Title',
        #                        'Genre', 'Language', 'Released', 'imdbRating']])
        scores[i].extend([meta[str(scores[i][0])]['omdb'][key]  for key in ['Title',
                                'Genre', 'imdbRating']])
        # scores[i][3] = ' '.join([genres_dict[i] for i in scores[i][3]])

    # indexes = ['id', 'score', 'Title', 'Genre', 'Language', 'Released', 'imdbRating']
    indexes = ['id', 'score', 'Title', 'Genre', 'imdbRating']
    table_dict = dict([(key, [i[idx] for i in scores]) for idx, key in enumerate(indexes)])
    table = pd.DataFrame(table_dict)
    return table


def render_header():
    st.write("""
        <p align="center"> 
            <img src="https://raw.githubusercontent.com/awarebayes/RecNN/master/res/logo%20big.png">
        </p>
        
        
        <p align="center"> 
        <iframe src="https://ghbtns.com/github-btn.html?user=awarebayes&repo=recnn&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>
        <iframe src="https://ghbtns.com/github-btn.html?user=awarebayes&repo=recnn&type=fork&count=true&size=large" frameborder="0" scrolling="0" width="158px" height="30px"></iframe>
        <iframe src="https://ghbtns.com/github-btn.html?user=awarebayes&type=follow&count=true&size=large" frameborder="0" scrolling="0" width="220px" height="30px"></iframe>
        </p>

        <p align="center"> 

        <a href='https://circleci.com/gh/awarebayes/RecNN'>
        <img src='https://circleci.com/gh/awarebayes/RecNN.svg?style=svg' alt='Documentation Status' />
        </a>

        <a href="https://codeclimate.com/github/awarebayes/RecNN/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/d3a06ffe45906969239d/maintainability" />            
        </a>

        <a href="https://colab.research.google.com/github/awarebayes/RecNN/">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" />
        </a>

        <a href='https://recnn.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/recnn/badge/?version=latest' alt='Documentation Status' />
        </a>

        </p>
        
        <p align="center"> 
            <b> Choose the page on the left sidebar to proceed </b>
        </p>

        <p align="center"> 
            This is my school project. It focuses on Reinforcement Learning for personalized news recommendation.
            The main distinction is that it tries to solve online off-policy learning with dynamically generated 
            item embeddings. I want to create a library with SOTA algorithms for reinforcement learning
            recommendation, providing the level of abstraction you like.
        </p>

        <p align="center">
            <a href="https://recnn.readthedocs.io">recnn.readthedocs.io</a>
        </p>
        
        ### Read the articles on medium!
        
        - Pretty much what you need to get started with this library if you know recommenders
          but don't know much about reinforcement learning:
        <p align="center"> 
           <a href="https://towardsdatascience.com/reinforcement-learning-ddpg-and-td3-for-news-recommendation-d3cddec26011">
                <img src="https://raw.githubusercontent.com/awarebayes/RecNN/master/res/article_1.png"  width="100%">
            </a>
        </p>
        
        - Top-K Off-Policy Correction for a REINFORCE Recommender System:
        <p align="center"> 
           <a href="https://towardsdatascience.com/top-k-off-policy-correction-for-a-reinforce-recommender-system-e34381dceef8">
                <img src="https://raw.githubusercontent.com/awarebayes/RecNN/master/res/article_2.png" width="100%">
            </a>
        </p>

    """, unsafe_allow_html=True)


@st.cache
def get_embs():
    return get_env().embeddings


def get_index():
    import faiss
    from sklearn.preprocessing import normalize
    # test indexes
    indexL2 = faiss.IndexFlatL2(128)
    indexIP = faiss.IndexFlatIP(128)
    indexCOS = faiss.IndexFlatIP(128)

    mov_mat = get_embs().numpy().astype('float32')
    indexL2.add(mov_mat)
    indexIP.add(mov_mat)
    indexCOS.add(normalize(mov_mat, axis=1, norm='l2'))
    return {'L2': indexL2, 'IP': indexIP, 'COS': indexCOS}


def main():
    st.sidebar.header('📰 recnn by @awarebayes 👨‍🔧')

    if st.sidebar.checkbox('Use cuda', True):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Get Started", "Recommend", "Test distance"])

    if page == "Get Started":
        render_header()

        st.markdown("""
            
            ## Available algorithms:
            
            | Algorithm                             | Paper                            | Code                       |
            |---------------------------------------|----------------------------------|----------------------------|
            | Deep Q Learning (PoC)                 | https://arxiv.org/abs/1312.5602  | examples/0. Embeddings/ 1.DQN |
            | Deep Deterministic Policy Gradients   | https://arxiv.org/abs/1509.02971 | examples/1.Vanilla RL/DDPG |
            | Twin Delayed DDPG (TD3)               | https://arxiv.org/abs/1802.09477 | examples/1.Vanilla RL/TD3  |
            | Soft Actor-Critic                     | https://arxiv.org/abs/1801.01290 | examples/1.Vanilla RL/SAC  |
            | Batch Constrained Q-Learning          | https://arxiv.org/abs/1812.02900 | examples/99.To be released/BCQ |
            | REINFORCE Top-K Off-Policy Correction | https://arxiv.org/abs/1812.02353 | examples/2. REINFORCE TopK |
        """)

        st.subheader("If you have cloned this repo, here is some stuff for you:")

        st.markdown(
            """
            ### Downloads
            - [MovieLens 20M (ratings.csv)](https://grouplens.org/datasets/movielens/20m/)
            - [My Movie Embeddings (ml20_pca128.pkl)](https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL)
            - [Meta (omdb.json)](https://drive.google.com/open?id=1t0LNCbqLjiLkAMFwtP8OIYU-zPUCNAjK)

            set RATINGSPATH, EMBEDPATH, METAPATH, MODELSPATH variables to these
            """
        )

    if page == "Recommend":

        st.header("Let's recommend something!")

        st.info("Upon the first opening the data will start loading."
                "\n Unfortunately there is no progress verbose in streamlit. Look in your console.")

        st.success('Data is loaded!')

        models = load_models(device)
        st.success('Models are loaded!')

        state, action, reward, next_state, done = get_batch(device)

        st.subheader('Here is a random batch sampled from testing environment:')
        if st.checkbox('Print batch info'):
            st.subheader('State')
            st.write(state)
            st.subheader('Action')
            st.write(action)
            st.subheader('Reward')
            st.write(reward.squeeze())

        st.subheader('(Optional) Select the state are getting the recommendations for')

        action_id = np.random.randint(0, state.size(0), 1)[0]
        action_id_manual = st.checkbox('Manually set state index')
        if action_id_manual:
            action_id = st.slider("Choose state index:", min_value=0, max_value=state.size(0))

        st.write('state:', state[action_id])

        algorithm = st.selectbox('Choose an algorithm', ('ddpg', 'td3'))
        metric = st.selectbox('Choose a metric', ('euclidean', 'cosine', 'correlation',
                                                  'canberra', 'minkowski', 'chebyshev',
                                                  'braycurtis', 'cityblock',))
        topk = st.slider("TOP K items to recommend:", min_value=1, max_value=30, value=7)

        dist = {'euclidean': distance.euclidean, 'cosine': distance.cosine,
                'correlation': distance.correlation, 'canberra': distance.canberra,
                'minkowski': distance.minkowski, 'chebyshev': distance.chebyshev,
                'braycurtis': distance.braycurtis, 'cityblock': distance.cityblock}

        action = models[algorithm].forward(state)

        st.markdown('**Recommendations for state with index {}**'.format(action_id))
        st.write(rank(action[action_id].detach().cpu().numpy(), dist[metric], topk))

        st.subheader('Pairwise distances for all actions in the batch:')
        st.pyplot(recnn.utils.pairwise_distances_fig(action))

    if page == "Test distance":
        st.header("Test distances")

        models = load_models(device)
        st.success('Models are loaded!')
        state, action, reward, next_state, done = get_batch(device)

        indexes = get_index()

        def query(index, action, k=20):
            D, I = index.search(action, k)
            return D, I

        def get_err(action, dist, k=5, euc=False):
            D, I = query(indexes[dist], action, k)
            if euc:
                D = D ** 0.5  # l2 -> euclidean
            mean = D.mean(axis=1).mean()
            std = D.std(axis=1).mean()
            return I, mean, std

        def get_action(model_name, action_id):
            gen_action = models[model_name].forward(state)
            gen_action = gen_action[action_id].detach().cpu().numpy()
            return gen_action

        st.subheader('(Optional) Select the state are getting the recommendations for')

        action_id = np.random.randint(0, state.size(0), 1)[0]
        action_id_manual = st.checkbox('Manually set state index')
        if action_id_manual:
            action_id = st.slider("Choose state index:", min_value=0, max_value=state.size(0))

        st.header('Metric')
        dist = st.selectbox('Select distance', ['L2', 'IP', 'COS'])

        ddpg_action = get_action('ddpg', action_id).reshape(1, -1)
        td3_action  = get_action('td3', action_id).reshape(1, -1)

        topk = st.slider("TOP K items to recommend:", min_value=1, max_value=30, value=10)

        ddpg_I, ddpg_mean, ddpg_std = get_err(ddpg_action, dist, topk, euc=True)
        td3_I, td3_mean, td3_std = get_err(td3_action, dist, topk, euc=True)

        # Mean Err
        st.subheader('Mean error')
        labels = ['DDPG', 'TD3']
        x_pos = np.arange(len(labels))
        CTEs = [ddpg_mean, td3_mean]
        error = [ddpg_std, td3_std]

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.bar(x_pos, CTEs, yerr=error, )
        ax.set_xticks(x_pos)
        ax.grid(False)
        ax.set_xticklabels(labels)
        ax.set_title(dist + ' error')
        ax.yaxis.grid(True)

        st.pyplot(fig)

        # Similarities
        st.header('Similarities')
        emb = get_embs()

        st.subheader('ddpg')
        st.pyplot(recnn.utils.pairwise_distances_fig(torch.tensor(emb[ddpg_I])))
        st.subheader('td3')
        st.pyplot(recnn.utils.pairwise_distances_fig(torch.tensor(emb[td3_I])))




if __name__ == "__main__":
    main()