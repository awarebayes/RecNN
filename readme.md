<p align="center"> 
<img src="./res/logo.png">
</p>

<p align="center"> 
This is my school project. It focuses on Reinforcement Learning for personalized news recommendation. The main distinction is that it tries to solve online off-policy learning with dynamically generated item embeddings. I want to create a library with SOTA algorithms for reinforcement learning recommendation, providing the level of abstraction you like.
</p>

### All in all the features can be summed up to:

- Abstract as you decide: you can import the entire algorthm (say DDPG) and tell it to ddpg.learn(batch), you can import networks and the learning function separately, create a custom loader for your task, or can define everything by yourself.

- Examples do not contain any of the junk code or workarounds: pure model defenition and the algorithm itself in one file. I wrote a couple of articles explaining how it functions

- The learning is built around sequential or frame environment that supports ML20M and like. Seq and Frame determine the length type of sequential data, seq is fully sequential dynamic size, while frame is just a static frame.

- State Representation module with various methods. For sequential state representation you can use: basic LSTM/RNN/GRU, 
Temporal Convolutional Networks, Echo State Networks and Chaos Free RNNs that are way faster that GRU.

- Pytorch 1.2 support with Tensorboard visualization

- New datasets will be added in the future.

> As of now, the package is in it's early stages and is not available for download on PyPi. Use git fetch instead.

## Medium Articles

The repo consists of two parts: the library (./recnn) and the playground (./examples)  where I explain how to work with certain things. 

- First article, the code is under notes/1. Vanilla RL/, it's very beginner friendly and covers basic Reinforcement Learning Approach:

<p align="center"> 
   <a href="https://towardsdatascience.com/reinforcement-learning-ddpg-and-td3-for-news-recommendation-d3cddec26011">
        <img src="./res/Article.png">
    </a>
</p>


## Algorithms that are/will be added:

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

## What is this?

This is my school project. It focuses on Reinforcement Learning for personalized news recommendation. The main distinction is that it tries to solve online off-policy learning with dynamically generated item embeddings. Also, there is no exploration, since we are working with a dataset. In the example section, I use Google's BERT on the ML20M dataset to extract contextual information from the movie description to form the latent vector representations. Later, you can use the same transformation on new, previously unseen items (hence, the embeddings are dynamically generated). If you don't want to bother with embeddings pipeline, I have a DQN embeddings generator as a proof of concept.


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
- [Movie Embeddings](https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL)
- [Misc Data](https://drive.google.com/open?id=1TclEmCnZN_Xkl3TfUXL5ivPYmLnIjQSu)
- [Parsed (omdb,tmdb)](https://drive.google.com/open?id=1t0LNCbqLjiLkAMFwtP8OIYU-zPUCNAjK)

## Models

- [Articles 1,2: DDPG, TD3, BCQ](https://drive.google.com/open?id=1a15mvtXZwOOSj9aQJNCxNlPMYREYYDxg)


