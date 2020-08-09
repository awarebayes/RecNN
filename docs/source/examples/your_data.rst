Working with your own data
==========================


Colab Version Here (clickable):

.. image:: https://colab.research.google.com/assets/colab-badge.svg
 :target: https://colab.research.google.com/drive/1xWX4JQvlcx3mizwL4gB0THEyxw6LsXTL

**Some things to know beforehand:**

When you load and preprocess data, all of the additional data preprocessing happens in the 'prepare_dataset'
function that you should pass. An example of that is in the your own data notebook. Also if you have inconsistent
indexes (i.e. movies index in MovieLens looks like [1, 3, 10, 20]), recnn handles in on its own, reducing
memory usage. There is no need to worry about mixing up indexes while preprocessing your own data.

Here is how default ML20M dataset is processed. Use this as a reference::

    def prepare_dataset(args_mut: DataFuncArgsMut, kwargs: DataFuncKwargs):
        # get args
        frame_size = kwargs.get('frame_size')
        key_to_id = args_mut.base.key_to_id
        df = args_mut.df

        # rating range mapped from [0, 5] to [-5, 5]
        df['rating'] = try_progress_apply(df['rating'], lambda i: 2 * (i - 2.5))
        # id's tend to be inconsistent and sparse so they are remapped here
        df['movieId'] = try_progress_apply(df['movieId'], lambda i: key_to_id.get(i))
        users = df[['userId', 'movieId']].groupby(['userId']).size()
        users = users[users > frame_size].sort_values(ascending=False).index

        if pd.get_type() == "modin":
            df = df._to_pandas() # pandas groupby is sync and doesnt affect performance
        ratings = df.sort_values(by='timestamp').set_index('userId').drop('timestamp', axis=1).groupby('userId')

        # Groupby user
        user_dict = {}

        def app(x):
            userid = int(x.index[0])
            user_dict[userid] = {}
            user_dict[userid]['items'] = x['movieId'].values
            user_dict[userid]['ratings'] = x['rating'].values

        try_progress_apply(ratings, app)

        args_mut.user_dict = user_dict
        args_mut.users = users

        return args_mut, kwargs

Look in reference/data/dataset_functions for further details. 

Toy Dataset
+++++++++++

The code below generates an artificial dataset::

    import pandas as pd
    import numpy as np
    import datetime
    import random
    import time

    def random_string_date():
      return datetime.datetime.strptime('{} {} {} {}'.format(random.randint(1, 366),
                                                             random.randint(0, 23),
                                                             random.randint(1, 59),
                                                              2019), '%j %H %M %Y').strftime("%m/%d/%Y, %H:%M:%S")

    def string_time_to_unix(s):
      return int(time.mktime(datetime.datetime.strptime(s, "%m/%d/%Y, %H:%M:%S").timetuple()))

    size = 100000
    n_emb = 1000
    n_usr = 1000
    mydf = pd.DataFrame({'book_id': np.random.randint(0, n_emb, size=size),
                         'reader_id': np.random.randint(1, n_usr, size=size),
                         'liked': np.random.randint(0, 2, size=size),
                         'when': [random_string_date() for i in range(size)]})
    my_embeddings = dict([(i, torch.tensor(np.random.randn(128)).float()) for i in range(n_emb)])
    mydf.head()

    # output:
       book_id  reader_id  liked                  when
          0      919        130      0  06/16/2019, 11:54:00
          1      850        814      1  11/29/2019, 12:35:00
          2      733        553      0  07/07/2019, 05:45:00
          3      902        695      1  02/03/2019, 10:29:00
          4      960        993      1  05/29/2019, 01:35:00

    # saving the data
    ! mkdir mydataset
    import pickle

    mydf.to_csv('mydataset/mydf.csv', index=False)
    with open('mydataset/myembeddings.pickle', 'wb') as handle:
        pickle.dump(my_embeddings, handle)


Writing custom preprocessing function
+++++++++++++++++++++++++++++++++++++

The following is a copy of the preprocessing function listed above to work with the toy dataset::

   def prepare_my_dataset(args_mut, kwargs):

        # get args
        frame_size = kwargs.get('frame_size')
        key_to_id = args_mut.base.key_to_id
        df = args_mut.df

        df['liked'] = df['liked'].apply(lambda a: (a - 1) * (1 - a) + a)
        df['when'] = df['when'].apply(string_time_to_unix)
        df['book_id'] = df['book_id'].apply(key_to_id.get)

        users = df[['reader_id', 'book_id']].groupby(['reader_id']).size()
        users = users[users > frame_size].sort_values(ascending=False).index

        # If using modin: pandas groupby is sync and doesnt affect performance
        # if pd.get_type() == "modin": df = df._to_pandas()  
        ratings = df.sort_values(by='when').set_index('reader_id').drop('when', axis=1).groupby('reader_id')

        # Groupby user
        user_dict = {}

        def app(x):
            userid = x.index[0]
            user_dict[int(userid)] = {}
            user_dict[int(userid)]['items'] = x['book_id'].values
            user_dict[int(userid)]['ratings'] = x['liked'].values

        ratings.apply(app)

        args_mut.user_dict = user_dict
        args_mut.users = users

        return args_mut, kwargs


Putting it all together
+++++++++++++++++++++++

Final touches::

    frame_size = 10
    batch_size = 25

    dirs = recnn.data.env.DataPath(
        base="/mydataset",
        embeddings="myembeddings.pickle",
        ratings="mydf.csv",
        cache="cache/frame_env.pkl", # cache will generate after you run
        use_cache=True # generally you want to save env after it runs
    )
    # pass prepare_my_dataset here
    env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size, prepare_dataset=prepare_my_dataset)

    # test function
    def run_tests():
        batch = next(iter(env.test_dataloader))
        loss = ddpg.update(batch, learn=False)
        return loss

    value_net  = recnn.nn.Critic(1290, 128, 256, 54e-2)
    policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)

    cuda = torch.device('cuda')
    ddpg = recnn.nn.DDPG(policy_net, value_net)
    ddpg = ddpg.to(cuda)
    plotter = recnn.utils.Plotter(ddpg.loss_layout, [['value', 'policy']],)

    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    %matplotlib inline

    plot_every = 3
    n_epochs = 2

    def learn():
        for epoch in range(n_epochs):
            for batch in tqdm(env.train_dataloader):
                loss = ddpg.update(batch, learn=True)
                plotter.log_losses(loss)
                ddpg.step()
                if ddpg._step % plot_every == 0:
                    clear_output(True)
                    print('step', ddpg._step)
                    test_loss = run_tests()
                    plotter.log_losses(test_loss, test=True)
                    plotter.plot_loss()
                if ddpg._step > 100:
                    return

    learn()
