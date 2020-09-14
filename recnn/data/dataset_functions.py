from .pandas_backend import pd
import numpy as np
from typing import List, Dict, Callable

"""   
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

"""


def try_progress_apply(dataframe, function):
    try:
        return dataframe.progress_apply(function)
    except AttributeError:
        return dataframe.apply(function)


# Plain args. Shouldn't be mutated
class DataFuncKwargs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def keys(self):
        return self.kwargs.keys()

    def get(self, name: str):
        if name not in self.kwargs:
            example = """
                # example on how to use kwargs:
                def prepare_dataset(args, args_mut):
                    args.set_kwarg('{}', your_value) # set kwargs for your functions here!
                    pipeline = [recnn.data.truncate_dataset, recnn.data.prepare_dataset]
                    recnn.data.build_data_pipeline(pipeline, args, args_mut)
            """
            raise AttributeError(
                "No kwarg with name {} found!\n{}".format(name, example.format(example))
            )
        return self.kwargs[name]

    def set(self, name: str, value):
        self.kwargs[name] = value


# Used for returning, arguments are mutable
class DataFuncArgsMut:
    def __init__(
        self, df, base, users: List[int], user_dict: Dict[int, Dict[str, np.ndarray]]
    ):
        self.base = base
        self.users = users
        self.user_dict = user_dict
        self.df = df


def prepare_dataset(args_mut: DataFuncArgsMut, kwargs: DataFuncKwargs):

    """
    Basic prepare dataset function. Automatically makes index linear, in ml20 movie indices look like:
    [1, 34, 123, 2000], recnn makes it look like [0,1,2,3] for you.
    """

    # get args
    frame_size = kwargs.get("frame_size")
    key_to_id = args_mut.base.key_to_id
    df = args_mut.df

    # rating range mapped from [0, 5] to [-5, 5]
    df["rating"] = try_progress_apply(df["rating"], lambda i: 2 * (i - 2.5))
    # id's tend to be inconsistent and sparse so they are remapped here
    df["movieId"] = try_progress_apply(df["movieId"], key_to_id.get)
    users = df[["userId", "movieId"]].groupby(["userId"]).size()
    users = users[users > frame_size].sort_values(ascending=False).index

    if pd.get_type() == "modin":
        df = df._to_pandas()  # pandas groupby is sync and doesnt affect performance
    ratings = (
        df.sort_values(by="timestamp")
        .set_index("userId")
        .drop("timestamp", axis=1)
        .groupby("userId")
    )

    # Groupby user
    user_dict = {}

    def app(x):
        userid = x.index[0]
        user_dict[userid] = {}
        user_dict[userid]["items"] = x["movieId"].values
        user_dict[userid]["ratings"] = x["rating"].values

    try_progress_apply(ratings, app)

    args_mut.user_dict = user_dict
    args_mut.users = users

    return args_mut, kwargs


def truncate_dataset(args_mut: DataFuncArgsMut, kwargs: DataFuncKwargs):
    """
    Truncate #items to reduce_items_to provided in kwargs
    """

    # here are adjusted n items to keep
    num_items = kwargs.get("reduce_items_to")
    df = args_mut.df

    counts = df["movieId"].value_counts().sort_values()
    to_remove = counts[:-num_items].index
    to_keep = counts[-num_items:].index
    to_keep_id = pd.get().Series(to_keep).apply(args_mut.base.key_to_id.get).values
    to_keep_mask = np.zeros(len(counts))
    to_keep_mask[to_keep_id] = 1

    args_mut.df = df.drop(df[df["movieId"].isin(to_remove)].index)

    key_to_id_new = {}
    id_to_key_new = {}
    count = 0

    for idx, i in enumerate(list(args_mut.base.key_to_id.keys())):
        if i in to_keep:
            key_to_id_new[i] = count
            id_to_key_new[idx] = i
            count += 1

    args_mut.base.embeddings = args_mut.base.embeddings[to_keep_mask]
    args_mut.base.key_to_id = key_to_id_new
    args_mut.base.id_to_key = id_to_key_new

    print(
        "action space is reduced to {} - {} = {}".format(
            num_items + len(to_remove), len(to_remove), num_items
        )
    )

    return args_mut, kwargs


def build_data_pipeline(
    chain: List[Callable], kwargs: DataFuncKwargs, args_mut: DataFuncArgsMut
):
    """
    Higher order function
    :param chain: array of callable
    :param **kwargs: any kwargs you like
    """
    for call in chain:
        # note: returned kwargs are not utilized to guarantee immutability
        args_mut, _ = call(args_mut, kwargs)
    return args_mut, kwargs
