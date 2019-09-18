<p align="center"> 
<img src="./res/logo big.png">
</p>
<p align="center"> 

<a href='https://recnn.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/recnn/badge/?version=latest' alt='Documentation Status' />
</a>

</p>
<p align="center"> 
This is my school project. It focuses on Reinforcement Learning for personalized news recommendation. The main distinction is that it tries to solve online off-policy learning with dynamically generated item embeddings. I want to create a library with SOTA algorithms for reinforcement learning recommendation, providing the level of abstraction you like.
</p>



### All in all the features can be summed up to:

- Abstract as you decide: you can import the entire algorithm (say DDPG) and tell it to ddpg.learn(batch), you can import networks and the learning function separately, create a custom loader for your task, or can define everything by yourself.

- Examples do not contain any of the junk code or workarounds: pure model definition and the algorithm itself in one file. I wrote a couple of articles explaining how it functions.

- The learning is built around sequential or frame environment that supports ML20M and like. Seq and Frame determine the length type of sequential data, seq is fully sequential dynamic size, while the frame is just a static frame.

- State Representation module with various methods. For sequential state representation, you can use basic LSTM/RNN/GRU, 
Temporal Convolutional Networks, Echo State Networks and Chaos Free RNNs that are way faster than GRU.

- Pytorch 1.2 support with Tensorboard visualization.

- New datasets will be added in the future.

- SOTA optimizers (RAdam, LookAhead, Ranger) come pre-packaged.

To find more about recnn, read the docs: 
 <a href="https://readthedocs.org/projects/recnn/">recnn.readthedocs.io</a> 

## Medium Articles

The repo consists of two parts: the library (./recnn) and the playground (./examples)  where I explain how to work with certain things. 

- The first article, the code is under notes/1. Vanilla RL/, it's very beginner-friendly and covers basic Reinforcement Learning Approach:

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
| Soft Actor-Critic                     | https://arxiv.org/abs/1801.01290 |examples/1.Vanilla RL/SAC         |
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

## Getting Started

<p align="center"> 
<a href="https://colab.research.google.com/drive/1qGjgLYuTpJTiW_P1pbs0zEgRa9BhZckw"><img src="./res/get_started.png"></a>
</p>

p.s. Image is clickable. here is [direct link](https://colab.research.google.com/drive/1qGjgLYuTpJTiW_P1pbs0zEgRa9BhZckw).

### Code snippet for simplest setup:

```
import torch
import recnn

env = recnn.data.env.FrameEnv('ml20_pca128.pkl','ml-20m/ratings.csv')

value_net  = recnn.nn.Critic(1290, 128, 256, 54e-2)
policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)

cuda = torch.device('cuda')
ddpg = recnn.nn.DDPG(policy_net, value_net)
ddpg = ddpg.to(cuda)

for batch in env.train_dataloader:
    ddpg.update(batch, learn=True)
```

### Installing

```
pip install git+git://github.com/awarebayes/RecNN.git



git clone https://github.com/awarebayes/RecNN
pip install ./RecNN

```

## Downloads
- [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
- [Movie Embeddings](https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL)
- [Misc Data](https://drive.google.com/open?id=1TclEmCnZN_Xkl3TfUXL5ivPYmLnIjQSu)
- [Parsed (omdb,tmdb)](https://drive.google.com/open?id=1t0LNCbqLjiLkAMFwtP8OIYU-zPUCNAjK)

## Models

- [Articles 1,2: DDPG, TD3, BCQ](https://drive.google.com/open?id=1a15mvtXZwOOSj9aQJNCxNlPMYREYYDxg)


## Citing
If you find RecNN useful for an academic publication, then please use the following BibTeX to cite it:

```
@misc{RecNN,
  author = {M Scherbina},
  title = {RecNN: RL Recommendation with PyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/awarebayes/RecNN}},
}
```

