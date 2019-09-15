Data
====
This module contains things to work with datasets. At the moment, utils are pretty messy and will be rewritten.



env
---

Main abstraction of the library for datasets is called environment, similar to how other reinforcement learning libraries name it. This interface is created to provide SARSA like input for your RL Models. When you are working with recommendation env, you have two choices: using static length inputs (say 10 items) or dynamic length time series with sequential encoders (many to one rnn). Static length is provided via FrameEnv, and dynamic length along with sequential state representation encoder is implemented in SeqEnv. Let's take a look at FrameEnv first:


.. automodule:: recnn.data.env
    :members:

Reference
+++++++++

.. autoclass:: UserDataset
    :members: __init__, __len__, __getitem__

.. autoclass:: Env
    :members: __init__

.. autoclass:: FrameEnv
    :members: __init__, train_batch, test_batch

.. autoclass:: SeqEnv
    :members: __init__, train_batch, test_batch
