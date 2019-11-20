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


dataset_functions
-----------------

What?
+++++

RecNN is designed to work with your dataflow.
Function that contain 'dataset' are needed to interact with environment.
The environment is provided via env.argument.
These functions can interact with env and set up some stuff how you like.
They are also designed to be argument agnostic

Basically you can stack them how you want.

To further illustrate this, let's take a look onto code sample from FrameEnv::

        class Env:
            def __init__(self, ...,
                 # look at this function provided here:
                 prepare_dataset=dataset_functions.prepare_dataset,
                 .....):

                self.user_dict = None
                self.users = None  # filtered keys of user_dict

                self.prepare_dataset(df=self.ratings, key_to_id=self.key_to_id,
                                     min_seq_size=min_seq_size, frame_size=min_seq_size, env=self)

                # after this call user_dict and users should be set to their values!

In reinforce example I further modify it to look like::

        def prepare_dataset(**kwargs):
            recnn.data.build_data_pipeline([recnn.data.truncate_dataset,
                                            recnn.data.prepare_dataset],
                                            reduce_items_to=5000, **kwargs)

Notice: prepare_dataset doesn't take **reduce_items_to** argument, but it is required in truncate_dataset.
As I previously mentioned RecNN is designed to be argument agnostic, meaning you provide some kwarg in the
build_data_pipeline function and it is passed down the function chain. If needed, it will be used. Otherwise ignored

.. automodule:: recnn.data.dataset_functions
    :members:

