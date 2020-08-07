from recnn.data.utils import make_items_tensor
from .pandas_backend import pd
"""     
    What?
    +++++
    
    Chain of responsibility pattern.
    https://refactoring.guru/design-patterns/chain-of-responsibility/python/example
    
    RecNN is designed to work with your data flow. 
    Function that contain 'dataset' are needed to interact with environment.
    The environment is provided via env.argument.
    These functions can interact with env and set up some stuff how you like.
    They are also designed to be argument agnostic
    
    Basically you can stack them how you want.
    
    To further illustrate this, let's take a look onto code sample from FrameEnv::
    
        class Env:
            def __init__(self, ...,
                 prepare_dataset=dataset_functions.prepare_dataset, # <- look at this function provided here
                 .....):
    
                self.user_dict = None
                self.users = None  # filtered keys of user_dict
            
                self.prepare_dataset(df=self.ratings, key_to_id=self.key_to_id,
                                     min_seq_size=min_seq_size, frame_size=min_seq_size, env=self)
                                         
                # after this call user_dict and users should be set to their values!
                
    In reinforce example I further modify it to look like::
    
        def prepare_dataset(**kwargs):
            recnn.data.build_data_pipeline([recnn.data.truncate_dataset,
                                            recnn.data.prepare_dataset], reduce_items_to=5000, **kwargs)
                                            
    Notice: prepare_dataset doesn't take **reduce_items_to** argument, but it is required in truncate_dataset.
    As I previously mentioned RecNN is designed to be argument agnostic, meaning you provide some kwarg in the  
    build_data_pipeline function, and it is passed down the function chain. If needed, it will be used. Otherwise, ignored  
"""

def try_progress_apply(dataframe, function):
    try:
        return dataframe.progress_apply(function)
    except AttributeError:
        return dataframe.apply(function)

def prepare_dataset(df, key_to_id, frame_size, env, sort_users=False, **kwargs):

    """
        Basic prepare dataset function. Automatically makes index linear, in ml20 movie indices look like:
        [1, 34, 123, 2000], recnn makes it look like [0,1,2,3] for you.
    """

    df['rating'] = try_progress_apply(df['rating'], lambda i: 2 * (i - 2.5))
    df['movieId'] = try_progress_apply(df['movieId'], lambda i: key_to_id.get(i))

    users = df[['userId', 'movieId']].groupby(['userId']).size()
    users = users[users > frame_size]
    if sort_users:
        users = users.sort_values(ascending=False)
    users = users.index

    if pd.get_type() == "modin":
        df = df._to_pandas()
    ratings = df.sort_values(by='timestamp').set_index('userId').drop('timestamp', axis=1).groupby('userId')

    # Groupby user
    user_dict = {}

    def app(x):
        userid = x.index[0]
        user_dict[int(userid)] = {}
        user_dict[int(userid)]['items'] = x['movieId'].values
        user_dict[int(userid)]['ratings'] = x['rating'].values

    try_progress_apply(ratings, app)

    env.user_dict = user_dict
    env.users = users

    return {'df': df, 'key_to_id': key_to_id,
            'frame_size': frame_size, 'env': env, 'sort_users': sort_users, **kwargs}


def truncate_dataset(df, key_to_id, frame_size, env, reduce_items_to, sort_users=False, **kwargs):
    """
        Truncate #items to num_items provided in the arguments
    """

    # here are adjusted n items to keep
    num_items = reduce_items_to

    to_remove = df['movieId'].value_counts().sort_values()[:-num_items].index
    to_keep = df['movieId'].value_counts().sort_values()[-num_items:].index
    to_remove_indices = df[df['movieId'].isin(to_remove)].index
    num_removed = len(to_remove)

    df.drop(to_remove_indices, inplace=True)

    for i in list(env.movie_embeddings_key_dict.keys()):
        if i not in to_keep:
            del env.movie_embeddings_key_dict[i]

    env.embeddings, env.key_to_id, env.id_to_key = make_items_tensor(env.movie_embeddings_key_dict)

    print('action space is reduced to {} - {} = {}'.format(num_items + num_removed, num_removed,
                                                           num_items))

    return {'df': df, 'key_to_id': env.key_to_id, 'env': env,
            'frame_size': frame_size, 'sort_users': sort_users, **kwargs}


def build_data_pipeline(chain, **kwargs):
    """
        Chain of responsibility pattern

        :param chain: array of callable
        :param **kwargs: any kwargs you like
    """

    kwargdict = kwargs
    for call in chain:
        kwargdict = call(**kwargdict)
    return kwargdict

