# Examples
This is the primary section that contains all of my experiments. As you can see, it is divided into subfolders, topics of which can vary. 

Topics:

0. Embeddings generation. This section covers basic python web crapping of RESTful APIs (OMDB, IMDB, TMDB) and feature engineering of the gathered data. The feature engineering part shows how to efficiently encode categories with PCA (Multiple Correspondence Analysis), assuming you have no missing values. Numerical data is processed with Probabilistic version of Principal Component Analysis (PCA) that can handle missing values, and do all sorts of fun stuff. I also attempt to visualize the embedding space using Uniform Manifold Approximation and Projection (UMAP). Text data is processed with Facebooks FairSeq RoBERTa, which I have finetuned on MNLI like dataset (Multi-Genre Natural Language Inference). I finetune it trying to predict whether two (or more) plots belong to the same movie. Then I downcasted all of the gathered data into 3 embedding types you can choose from: UMAP, PCA (recommended) and AutoEncoder.

1. Vanilla Reinforcement Learning: this section contains my implementations for basic RL Actor-Critic algorithms.

2. Reinforce TOP K OffPolicy Correction.

 I took inspiration from Higgsfield's code, but I have rewritten most of the stuff in order to achieve
  the abstract and clear look of the code. 