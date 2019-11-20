.. image:: https://github.com/awarebayes/RecNN/raw/master/res/logo%20big.png
    :align: center

Welcome to recnn's documentation!
========================================

What
++++

This is my school project. It focuses on Reinforcement Learning for personalized
news recommendation. The main distinction is that it tries to solve online off-policy
learning with dynamically generated item embeddings. Also, there is no exploration,
since we are working with a dataset. In the example section, I use Google's BERT on
the ML20M dataset to extract contextual information from the movie description to form
the latent vector representations. Later, you can use the same transformation on new,
previously unseen items (hence, the embeddings are dynamically generated). If you don't
want to bother with embeddings pipeline, I have a DQN embeddings generator as a proof
of concept.

Getting Started
+++++++++++++++

There are a couple of ways you can get started. The most straightforward is to clone and go to the examples section.
You can also use Google Colab or Gradient Experiment.


How parameters should look like::

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


.. toctree::
   :maxdepth: 3
   :caption: Tutorials:

      Tutorials <examples/examples>


.. toctree::
   :maxdepth: 2
   :caption: Reference:

      NN <nn>

      Data <data>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

