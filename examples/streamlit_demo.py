import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import pickle
import json
import copy
import pandas as pd
import random
from tqdm.auto import tqdm

import torch
from scipy.spatial import distance

# == recnn ==
import sys

sys.path.append("../")
import recnn


tqdm.pandas()

# constants
ML20MPATH = "../data/ml-20m/"
MODELSPATH = "../models/"
DATAPATH = "../data/streamlit/"
SHOW_TOPN_MOVIES = (
    200  # recommend me a movie. show only top ... movies, higher values lead to slow ux
)

# disable it if you get an error
from jupyterthemes import jtplot

jtplot.style(theme="grade3")


def render_header():
    st.write(
        """
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

        ### 📚 Read the articles on medium!

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

    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """

        ### 🤖 You can play with these (more will be implemented):

        | Algorithm                             | Paper                            | Code                       |
        |---------------------------------------|----------------------------------|----------------------------|
        | Deep Deterministic Policy Gradients   | https://arxiv.org/abs/1509.02971 | examples/1.Vanilla RL/DDPG |
        | Twin Delayed DDPG (TD3)               | https://arxiv.org/abs/1802.09477 | examples/1.Vanilla RL/TD3  |
        | Soft Actor-Critic                     | https://arxiv.org/abs/1801.01290 | examples/1.Vanilla RL/SAC  |
        | REINFORCE Top-K Off-Policy Correction | https://arxiv.org/abs/1812.02353 | examples/2. REINFORCE TopK |
    """
    )


@st.cache
def load_mekd():
    return pickle.load(open(DATAPATH + "mekd.pkl", "rb"))


def get_batch(device):
    # gets a random batch using cached load
    @st.cache
    def load_batch():
        return pickle.load(open(DATAPATH + "batch.pkl", "rb"))

    # todo remove randomness
    return [i.to(device) for i in random.choice(load_batch())]


def get_embeddings():
    movie_embeddings_key_dict = load_mekd()
    movies_embeddings_tensor, key_to_id, id_to_key = recnn.data.utils.make_items_tensor(
        movie_embeddings_key_dict
    )
    return movies_embeddings_tensor, key_to_id, id_to_key


@st.cache
def load_omdb_meta():
    return json.load(open(DATAPATH + "omdb.json"))


def load_models(device):
    ddpg = recnn.nn.models.Actor(1290, 128, 256).to(device)
    td3 = recnn.nn.models.Actor(1290, 128, 256).to(device)

    ddpg.load_state_dict(
        torch.load(MODELSPATH + "ddpg_policy.model", map_location=device)
    )
    td3.load_state_dict(
        torch.load(MODELSPATH + "td3_policy.model", map_location=device)
    )
    return {"ddpg": ddpg, "td3": td3}


@st.cache
def load_links():
    return pd.read_csv(ML20MPATH + "links.csv", index_col="tmdbId")


@st.cache
def get_mov_base():
    links = load_links()
    movies_embeddings_tensor, key_to_id, id_to_key = get_embeddings()
    meta = load_omdb_meta()

    popular = pd.read_csv(DATAPATH + "movie_counts.csv")[:SHOW_TOPN_MOVIES]
    st.write(popular["id"])
    mov_base = {}

    for i, k in list(meta.items()):
        tmdid = int(meta[i]["tmdbId"])
        if tmdid > 0 and popular["id"].isin([i]).any():
            movieid = pd.to_numeric(links.loc[tmdid]["movieId"])
            if isinstance(movieid, pd.Series):
                continue
            mov_base[int(movieid)] = meta[i]["omdb"]["Title"]

    return mov_base


def get_index():
    import faiss
    from sklearn.preprocessing import normalize

    # test indexes
    indexL2 = faiss.IndexFlatL2(128)
    indexIP = faiss.IndexFlatIP(128)
    indexCOS = faiss.IndexFlatIP(128)

    mov_mat, _, _ = get_embeddings()
    mov_mat = mov_mat.numpy().astype("float32")
    indexL2.add(mov_mat)
    indexIP.add(mov_mat)
    indexCOS.add(normalize(mov_mat, axis=1, norm="l2"))
    return {"L2": indexL2, "IP": indexIP, "COS": indexCOS}


def rank(gen_action, metric, k):
    scores = []
    movie_embeddings_key_dict = load_mekd()
    meta = load_omdb_meta()

    for i in movie_embeddings_key_dict.keys():
        if i == 0 or i == "0":
            continue
        scores.append([i, metric(movie_embeddings_key_dict[i], gen_action)])
    scores = list(sorted(scores, key=lambda x: x[1]))
    scores = scores[:k]
    ids = [i[0] for i in scores]
    for i in range(k):
        scores[i].extend(
            [
                meta[str(scores[i][0])]["omdb"][key]
                for key in ["Title", "Genre", "imdbRating"]
            ]
        )
    indexes = ["id", "score", "Title", "Genre", "imdbRating"]
    table_dict = dict(
        [(key, [i[idx] for i in scores]) for idx, key in enumerate(indexes)]
    )
    table = pd.DataFrame(table_dict)
    return table


@st.cache
def load_reinforce():
    indexes = pickle.load(open(DATAPATH + "reinforce_indexes.pkl", "rb"))
    state = pickle.load(open(DATAPATH + "reinforce_state.pkl", "rb"))
    return state, indexes


def main():
    st.sidebar.header("📰 recnn by @awarebayes 👨‍🔧")

    if st.sidebar.checkbox("Use cuda", torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    st.sidebar.subheader("Choose a page to proceed:")
    page = st.sidebar.selectbox(
        "",
        [
            "🚀 Get Started",
            "📽 ️Recommend me a movie",
            "🔨 Test Recommendation",
            "⛏️ Test Diversity",
            "🤖 Reinforce Top K",
        ],
    )

    if page == "🚀 Get Started":
        render_header()

        st.subheader("If you have cloned this repo, here is some stuff for you:")

        st.markdown(
            """
            📁 **Downloads** + change the **constants**, so they point to this unpacked folder:
            
            - [Models](https://drive.google.com/file/d/1goGa15XZmDAp2msZvRi2v_1h9xfmnhz7/view?usp=sharing)
             **= MODELSPATH**
            - [Data for Streamlit Demo](https://drive.google.com/file/d/1nuhHDdC4mCmiB7g0fmwUSOh1jEUQyWuz/view?usp=sharing)
             **= DATAPATH**
            - [ML20M Dataset](https://grouplens.org/datasets/movielens/20m/)
             **= ML20MPATH**
             
            p.s. ml20m is only needed for links.csv, I couldn't include it in my streamlit data because of copyright.
            This is all the data you need.
            """
        )

    if page == "🔨 Test Recommendation":

        st.header("Test the Recommendations")

        st.info(
            "Upon the first opening the data will start loading."
            "\n Unfortunately there is no progress verbose in streamlit. Look in your console."
        )

        st.success("Data is loaded!")

        models = load_models(device)
        st.success("Models are loaded!")

        state, action, reward, next_state, done = get_batch(device)

        st.subheader("Here is a random batch sampled from testing environment:")
        if st.checkbox("Print batch info"):
            st.subheader("State")
            st.write(state)
            st.subheader("Action")
            st.write(action)
            st.subheader("Reward")
            st.write(reward.squeeze())

        st.subheader("(Optional) Select the state are getting the recommendations for")

        action_id = np.random.randint(0, state.size(0), 1)[0]
        action_id_manual = st.checkbox("Manually set state index")
        if action_id_manual:
            action_id = st.slider(
                "Choose state index:", min_value=0, max_value=state.size(0)
            )

        st.write("state:", state[action_id])

        algorithm = st.selectbox("Choose an algorithm", ("ddpg", "td3"))
        metric = st.selectbox(
            "Choose a metric",
            (
                "euclidean",
                "cosine",
                "correlation",
                "canberra",
                "minkowski",
                "chebyshev",
                "braycurtis",
                "cityblock",
            ),
        )
        topk = st.slider(
            "TOP K items to recommend:", min_value=1, max_value=30, value=7
        )

        dist = {
            "euclidean": distance.euclidean,
            "cosine": distance.cosine,
            "correlation": distance.correlation,
            "canberra": distance.canberra,
            "minkowski": distance.minkowski,
            "chebyshev": distance.chebyshev,
            "braycurtis": distance.braycurtis,
            "cityblock": distance.cityblock,
        }

        action = models[algorithm].forward(state)

        st.markdown("**Recommendations for state with index {}**".format(action_id))
        st.write(rank(action[action_id].detach().cpu().numpy(), dist[metric], topk))

        st.subheader("Pairwise distances for all actions in the batch:")
        st.pyplot(recnn.utils.pairwise_distances_fig(action))

    if page == "⛏️ Test Diversity":
        st.header("Test the Distances (diversity and pinpoint accuracy)")

        models = load_models(device)
        st.success("Models are loaded!")
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

        st.subheader("(Optional) Select the state are getting the recommendations for")

        action_id = np.random.randint(0, state.size(0), 1)[0]
        action_id_manual = st.checkbox("Manually set state index")
        if action_id_manual:
            action_id = st.slider(
                "Choose state index:", min_value=0, max_value=state.size(0)
            )

        st.header("Metric")
        dist = st.selectbox("Select distance", ["L2", "IP", "COS"])

        ddpg_action = get_action("ddpg", action_id).reshape(1, -1)
        td3_action = get_action("td3", action_id).reshape(1, -1)

        topk = st.slider(
            "TOP K items to recommend:", min_value=1, max_value=30, value=10
        )

        ddpg_I, ddpg_mean, ddpg_std = get_err(ddpg_action, dist, topk, euc=True)
        td3_I, td3_mean, td3_std = get_err(td3_action, dist, topk, euc=True)

        # Mean Err
        st.subheader("Mean error")
        st.markdown(
            """
        How close are we to the actual movie embedding? 
        
        The closer the better, although higher error may
        produce more diverse recommendations.
        """
        )
        labels = ["DDPG", "TD3"]
        x_pos = np.arange(len(labels))
        CTEs = [ddpg_mean, td3_mean]
        error = [ddpg_std, td3_std]

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.bar(
            x_pos,
            CTEs,
            yerr=error,
        )
        ax.set_xticks(x_pos)
        ax.grid(False)
        ax.set_xticklabels(labels)
        ax.set_title(dist + " error")
        ax.yaxis.grid(True)

        st.pyplot(fig)

        # Similarities
        st.header("Similarities")
        emb, _, _ = get_embeddings()

        st.markdown(
            "Heatmap of correlation similarities (Grammarian Product of actions)"
            "\n\n"
            "Higher = mode diverse, lower = less diverse. You decide what is better..."
        )

        st.subheader("ddpg")
        st.pyplot(recnn.utils.pairwise_distances_fig(torch.tensor(emb[ddpg_I])))
        st.subheader("td3")
        st.pyplot(recnn.utils.pairwise_distances_fig(torch.tensor(emb[td3_I])))

    if page == "📽 ️Recommend me a movie":
        st.header("📽 ️Recommend me a movie")
        st.markdown(
            """
        **Now, this is probably why you came here. Let's get you some movies suggested**
        
        You need to choose 10 movies in the bar below by typing their titles.
        Due to the client side limitations, I am only able to display top 200 movies.
        P.S. you can type to search
        """
        )

        mov_base = get_mov_base()
        mov_base_by_title = {v: k for k, v in mov_base.items()}
        movies_chosen = st.multiselect("Choose 10 movies", list(mov_base.values()))
        st.markdown(
            "**{} chosen {} to go**".format(len(movies_chosen), 10 - len(movies_chosen))
        )

        if len(movies_chosen) > 10:
            st.error(
                "Please select exactly 10 movies, you have selected {}".format(
                    len(movies_chosen)
                )
            )
        if len(movies_chosen) == 10:
            st.success("You have selected 10 movies. Now let's rate them")
        else:
            st.info("Please select 10 movies in the input above")

        if len(movies_chosen) == 10:
            st.markdown("### Rate each movie from 1 to 10")
            ratings = dict(
                [
                    (i, st.number_input(i, min_value=1, max_value=10, value=5))
                    for i in movies_chosen
                ]
            )
            # st.write('for debug your ratings are:', ratings)

            ids = [mov_base_by_title[i] for i in movies_chosen]
            # st.write('Movie indexes', list(ids))
            embs = load_mekd()
            state = torch.cat(
                [
                    torch.cat([embs[i] for i in ids]),
                    torch.tensor(list(ratings.values())).float() - 5,
                ]
            )
            st.write("your state", state)
            state = state.to(device).squeeze(0)

            models = load_models(device)
            algorithm = st.selectbox("Choose an algorithm", ("ddpg", "td3"))

            metric = st.selectbox(
                "Choose a metric",
                (
                    "euclidean",
                    "cosine",
                    "correlation",
                    "canberra",
                    "minkowski",
                    "chebyshev",
                    "braycurtis",
                    "cityblock",
                ),
            )

            dist = {
                "euclidean": distance.euclidean,
                "cosine": distance.cosine,
                "correlation": distance.correlation,
                "canberra": distance.canberra,
                "minkowski": distance.minkowski,
                "chebyshev": distance.chebyshev,
                "braycurtis": distance.braycurtis,
                "cityblock": distance.cityblock,
            }

            topk = st.slider(
                "TOP K items to recommend:", min_value=1, max_value=30, value=7
            )
            action = models[algorithm].forward(state)

            st.subheader("The neural network thinks you should watch:")
            st.write(rank(action[0].detach().cpu().numpy(), dist[metric], topk))

    if page == "🤖 Reinforce Top K":
        st.title("🤖 Reinforce Top K")
        st.markdown(
            "**Reinforce is a discrete state algorithm, meaning a lot of metrics (i.e. error, diversity test) "
            "won't be possible. **"
        )
        st.subheader("This page is under construction")


if __name__ == "__main__":
    main()
