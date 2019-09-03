<p align="center"> 
<img src="./res/logo.png">
</p>

This is my school project. It focuses on Reinforcement Learning for personalized news recommendation. The main distinction is that it tries to solve online off-policy learning with dynamically generated item embeddings. Also, there is no exploration, since we are working with a dataset. In the example section, I use Google's BERT on the ML20M dataset to extract contextual information from the movie description to form the latent vector representations. Later, you can use the same transformation on new, previously unseen items (hence, the embeddings are dynamically generated). If you don't want to bother with embeddings pipeline, I have a DQN embeddings generator as a proof of concept.

The repo consists of two parts: the library (./recnn) and the playground (./examples)  where I explain how to work with certain things. I wrote a couple of articles explaining how it functions. 

- First article, the code is under notes/1. Vanilla RL/, it's very beginner friendly and covers basic Reinforcement Learning Approach:

<p align="center"> 
   <a href="https://towardsdatascience.com/reinforcement-learning-ddpg-and-td3-for-news-recommendation-d3cddec26011">
        <img src="./res/Article.png">
    </a>
</p>


 <h2> Algorithms that are/will be added: </h2> 

<p align="center"> 
    
| Algorithm                             | Paper                            | Code                       |
|---------------------------------------|----------------------------------|----------------------------|
| Deep Q Learning (PoC)                 | https://arxiv.org/abs/1312.5602  | examples/0. Embeddings/ 1.DQN                  |
| Deep Deterministic Policy Gradients   | https://arxiv.org/abs/1509.02971 | examples/1.Vanilla RL/DDPG |
| Twin Delayed DDPG (TD3)               | https://arxiv.org/abs/1802.09477 | examples/1.Vanilla RL/TD3  |
| Soft Actor Critic                     | https://arxiv.org/abs/1801.01290 |examples/1.Vanilla RL/SAC         |
| Batch Constrained Q-Learning          | https://arxiv.org/abs/1812.02900 | examples/2.BCQ/BCQ Pytorch |
| REINFORCE Top-K Off-Policy Correction | https://arxiv.org/abs/1509.02971 | WIP                        |

</p>

***
 ### [My Trello with useful papers](https://trello.com/b/wnor4IZf/recnn)
*** 
### Repos I used code from:

- Sfujim's [BCQ](https://github.com/sfujim/BCQ)
- LiyuanLucasLiu [Radam](https://github.com/LiyuanLucasLiu/RAdam)
- Higgsfield's [RL Adventure 2](https://github.com/higgsfield/RL-Adventure-2)

## Dataset Description
This project is built for MovieLens 20M dataset. But you can use it with your data. You will need:
1. Embeddings in {item_id: numpy.ndarray} format
2. CSV dataset: user_id, item_id, rating, timestamp

If you dont want to bother generating embeddings, use DQN in the 0./ Embeddings section.
I also have parsed all the movies in the '/links.csv' to get all auxiliary data from TMDB/IMDB. Text information was fed into Google's BERT/ OpenAI GPT2 models to get text embeddings. If you want to download anything, the links are down the description. 

## Misc Data

Everything of the misc sort is in the 'Misc Data' you can download in the downloads section, featuring all sorts of auxiliary stuff. Primarily it is movie info. If you don't want to use the embeddings, or just want to have some debug info/data for application this is what you need.

All text information is located in texts_bert.p / texts_gpt2.p in a dict {movie_id: numpy_array} format.

All of the categorical features had been label encoded, numerical standardized.

Here is an example of how the movie information looks like:

```python
{'adult': False,
 'collection': 210,
 'genres': [14, 1, 11],
 'original_language': 0,
 'popularity': 5.218749755002595,
 'production_companies': [96],
 'production_countries': [0],
 'release_year': 1995,
 'release_month': 10,
 'revenue': 4.893588591235185,
 'runtime': -0.5098445413830461,
 'spoken_languages': [0],
 'title': 'Toy Story',
 'vote_average': 1.2557064312220563,
 'vote_count': 1.8032194192281197,
 'budget': 1.1843770075921112,
 'revenue_d': 5.626649137875692}
```

## Getting started:

1. Download the ml20m dataset and the movie embeddings
2. Clone this repo
3. Infos_pca128.pytorch (embeddings) into the RecNN/data folder
4. Run notes/3. DDPG and see the results

## TD3 results

Here you can see the training process of the network:

<p align="center"> 
<img src="./res/Losses.png">
</p>

Here is a pairwise similarity matrix of real (first image) and generated (second image) actions (movies)


<p align="center"> 
    <img src="./res/real_dist.png">
</p>

<p align="center"> 
    <img src="./res/gen_dist.png">
</p>

It doesn't seem to overfit much. Here you can see the Kernel Density Estimation for Autoencoder Reconstruction scores. I use it as an anomaly detection metric. (Wasserstein Distance = ~50)

<p align="center"> 
<img src="./res/Anomaly_Detection.png">
</p>

 # Downloads
- [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
- [Movie Embeddings](https://drive.google.com/open?id=1kTyu05ZmtP2MA33J5hWdX8OyUYEDW4iI)
- [Misc Data](https://drive.google.com/open?id=1TclEmCnZN_Xkl3TfUXL5ivPYmLnIjQSu)
- [Metadata for predictions](https://drive.google.com/open?id=1xjVI4uVQGsQ7tjOJ3594ZXmAEC_6yX0e)

## Models

- [Articles 1,2: DDPG, TD3, BCQ](https://drive.google.com/open?id=1a15mvtXZwOOSj9aQJNCxNlPMYREYYDxg)
 
 **What is the size of ...?**
 
| Name       | Dimensions  | Base Type |
|------------|----------------|-----------|
| State      | 1290           | float     | 
| Action     | 128            | float     | 
| Reward     | 1              | int8      | 
| Next_State | 1290           | float     | 
| Done       | 1              | bool      | 

P.S. all types are downcasted to float32 in the PyTorch backend.

## Medium Articles (WIP)
I wrote some medium articles explaining how this works: 

<p align="center"> 
   <a href="https://towardsdatascience.com/reinforcement-learning-ddpg-and-td3-for-news-recommendation-d3cddec26011">
        <img src="./res/Article.png">
    </a>
</p>

<p align="center"> 
   <a href="https://towardsdatascience.com/deep-reinforcement-learning-for-news-recommendation-part-1-architecture-5741b1a6ed56">
        <img src="./res/Article old.png">
    </a>
</p>

