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

    def prepare_ml20m_dataset(df, key_to_id, frame_size, sort_users=False):
        df['rating'] = df['rating'].progress_apply(lambda i: 2 * (i - 2.5))
        df['movieId'] = df['movieId'].progress_apply(key_to_id.get)
        users = df[['userId', 'movieId']].groupby(['userId']).size()
        users = users[users > frame_size]
        if sort_users:
            users = users.sort_values(ascending=False)
        users = users.index
        ratings = df.sort_values(by='timestamp').set_index('userId')
        ratings = ratings.drop('timestamp', axis=1).groupby('userId')

        # Groupby user
        user_dict = {}

        def app(x):
            userid = x.index[0]
            user_dict[int(userid)] = {}
            user_dict[int(userid)]['items'] = x['movieId'].values
            user_dict[int(userid)]['ratings'] = x['rating'].values

        ratings.progress_apply(app)
        return user_dict, users


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

    def prepare_my_dataset(df, key_to_id, frame_size, sort_users=False):
        # transform [0 1] -> [-1 1]
        # you can also choose not use progress_apply here

        df['liked'] = df['liked'].progress_apply(lambda a: (a - 1) * (1 - a) + a)
        df['when'] = df['when'].progress_apply(string_time_to_unix)
        df['book_id'] = df['book_id'].progress_apply(key_to_id.get)
        users = df[['reader_id', 'book_id']].groupby(['reader_id']).size()
        users = users[users > frame_size]
        if sort_users:
            users = users.sort_values(ascending=False)

        users = users.index
        ratings = df.sort_values(by='when').set_index('reader_id')
        ratings = ratings.drop('when', axis=1).groupby('reader_id')

        # Groupby user
        user_dict = {}

        def app(x):
            userid = x.index[0]
            user_dict[int(userid)] = {}
            user_dict[int(userid)]['items'] = x['book_id'].values
            user_dict[int(userid)]['ratings'] = x['liked'].values

        ratings.progress_apply(app)
        return user_dict, users

Putting it all together
+++++++++++++++++++++++

Final touches::

    frame_size = 10
    batch_size = 25

    env = recnn.data.env.FrameEnv('mydataset/myembeddings.pickle', 'mydataset/mydf.csv',
                                  frame_size, batch_size, prepare_dataset=prepare_my_dataset) # <- ! pass YOUR function here

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
