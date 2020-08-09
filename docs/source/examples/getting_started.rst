Getting Started with recnn
==========================

Colab Version Here (clickable):

.. image:: https://colab.research.google.com/assets/colab-badge.svg
 :target: https://colab.research.google.com/drive/1xWX4JQvlcx3mizwL4gB0THEyxw6LsXTL


Offline example is in: RecNN/examples/[Library Basics]/1. Getting Started.ipynb

Let's do some imports::

    import recnn

    import recnn
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm

    tqdm.pandas()

    from jupyterthemes import jtplot
    jtplot.style(theme='grade3')

Environments
++++++++++++
Main abstraction of the library for datasets is called environment, similar to how other reinforcement learning libraries name it. This interface is created to provide SARSA like input for your RL Models. When you are working with recommendation env, you have two choices: using static length inputs (say 10 items) or dynamic length time series with sequential encoders (many to one rnn). Static length is provided via FrameEnv, and dynamic length along with sequential state representation encoder is implemented in SeqEnv. Let’s take a look at FrameEnv first:

In order to initialize an env, you need to provide embeddings and ratings directories::

    frame_size = 10
    batch_size = 25
    # embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
    dirs = recnn.data.env.DataPath(
        base="../../../data/",
        embeddings="embeddings/ml20_pca128.pkl",
        ratings="ml-20m/ratings.csv",
        cache="cache/frame_env.pkl", # cache will generate after you run
        use_cache=True
    )
    env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

    train = env.train_batch()
    test = env.train_batch()
    state, action, reward, next_state, done = recnn.data.get_base_batch(train, device=torch.device('cpu'))

    print(state)

    # State
    tensor([[  5.4261,  -4.6243,   2.3351,  ...,   3.0000,   4.0000,   1.0000],
        [  6.2052,  -1.8592,  -0.3248,  ...,   4.0000,   1.0000,   4.0000],
        [  3.2902,  -5.0021, -10.7066,  ...,   1.0000,   4.0000,   2.0000],
        ...,
        [  3.0571,  -4.1390,  -2.7344,  ...,   3.0000,  -3.0000,  -1.0000],
        [  0.8177,  -7.0827,  -0.6607,  ...,  -3.0000,  -1.0000,   3.0000],
        [  9.0742,   0.3944,  -6.4801,  ...,  -1.0000,   3.0000,  -1.0000]])

Recommending
++++++++++++

Let's initialize main networks, and recommend something! ::

    value_net  = recnn.nn.Critic(1290, 128, 256, 54e-2)
    policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)

    recommendation = policy_net(state)
    value = value_net(state, recommendation)
    print(recommendation)
    print(value)

    # Output:

    tensor([[ 1.5302, -2.3658,  1.6439,  ...,  0.1297,  2.2236,  2.9672],
        [ 0.8570, -1.3491, -0.3350,  ..., -0.8712,  5.8390,  3.0899],
        [-3.3727, -3.6797, -3.9109,  ...,  3.2436,  1.2161, -1.4018],
        ...,
        [-1.7834, -0.4289,  0.9808,  ..., -2.3487, -5.8386,  3.5981],
        [ 2.3813, -1.9076,  4.3054,  ...,  5.2221,  2.3165, -0.0192],
        [-3.8265,  1.8143, -1.8106,  ...,  3.3988, -3.1845,  0.7432]],
       grad_fn=<AddmmBackward>)
    tensor([[-1.0065],
            [ 0.3728],
            [ 2.1063],
            ...,
            [-2.1382],
            [ 0.3330],
            [ 5.4069]], grad_fn=<AddmmBackward>)

Algo classes
++++++++++++

Algo is a high level abstraction for an RL algorithm. You need two networks
(policy and value) in order to initialize it. Later on you can tweak parameters
and stuff in the algo itself.

Important: you can set writer to torch.SummaryWriter and get the debug output
Tweak how you want::

    ddpg = recnn.nn.DDPG(policy_net, value_net)
    print(ddpg.params)
    ddpg.params['gamma'] = 0.9
    ddpg.params['policy_step'] = 3
    ddpg.optimizers['policy_optimizer'] = torch.optim.Adam(ddpg.nets['policy_net'], your_lr)
    ddpg.writer = torch.utils.tensorboard.SummaryWriter('./runs')
    ddpg = ddpg.to(torch.device('cuda'))

ddpg.loss_layout is also handy, it allows you to see how the loss should look like ::

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
    ddpg.writer = SummaryWriter(dir='./runs')

    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    %matplotlib inline

    plot_every = 50
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
                if ddpg._step > 1000:
                    return

    learn()

Update Functions
++++++++++++++++

Basically, the Algo class is a high level wrapper around the update function. The code for that is pretty messy,
so if you want to check it out, I explained it in the colab notebook linked at the top.
