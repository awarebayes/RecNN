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

RecNN is designed to work with your data flow. 

Set kwargs in the beginning of prepare_dataset function.
Kwargs you set are immutable.

args_mut are mutable arguments, you can access the following:
    base: data.EnvBase, df: DataFrame, users: List[int],
    user_dict: Dict[int, Dict[str, np.ndarray]

Access args_mut and modify them in functions defined by you.
Best to use function chaining with build_data_pipeline.

recnn.data.prepare_dataset is a function that is used by default in Env.__init__
But sometimes you want some extra. I have also predefined truncate_dataset.
This function truncates the number of items to specified one.
In reinforce example I modify it to look like::
        
    def prepare_dataset(args_mut, kwargs):
        kwargs.set('reduce_items_to', num_items) # set kwargs for your functions here!
        pipeline = [recnn.data.truncate_dataset, recnn.data.prepare_dataset]
        recnn.data.build_data_pipeline(pipeline, kwargs, args_mut)
        
    # embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
    env = recnn.data.env.FrameEnv('..',
                                '...', frame_size, batch_size,
                                embed_batch=embed_batch, prepare_dataset=prepare_dataset,
                                num_workers=0)

.. automodule:: recnn.data.dataset_functions
    :members:

