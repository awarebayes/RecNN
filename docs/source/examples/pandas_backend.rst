Using Pandas Backends
==========================


RecNN supports different types of pandas backends for faster data loading/processing in and out of core


Pandas is your default backend::

    # but you can also set it directly:
    recnn.pd.set("pandas")
    frame_size = 10
    batch_size = 25
    dirs = recnn.data.env.DataPath(
        base="../../../data/",
        embeddings="embeddings/ml20_pca128.pkl",
        ratings="ml-20m/ratings.csv",
        cache="cache/frame_env.pkl", # cache will generate after you run
        use_cache=False # disable for testing purposes
    )

    %%time
    env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

    # Output:
    100%|██████████| 20000263/20000263 [00:13<00:00, 1469488.15it/s]
    100%|██████████| 20000263/20000263 [00:15<00:00, 1265183.17it/s]
    100%|██████████| 138493/138493 [00:06<00:00, 19935.53it/s]
    CPU times: user 41.6 s, sys: 1.89 s, total: 43.5 s
    Wall time: 43.5 s


IP.S. nstall Modin `here
<https://github.com/modin-project/modin/>`_ , it is not installed via RecNN's deps

You can also use modin  with Dask / Ray.

Here is a little Ray example::

    import os
    import ray

    if ray.is_initialized():
        ray.shutdown()
    os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
    ray.init(num_cpus=10) # adjust for your liking
    recnn.pd.set("modin")
    %%time
    env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

    100%|██████████| 138493/138493 [00:07<00:00, 18503.97it/s]
    CPU times: user 12 s, sys: 2.06 s, total: 14 s
    Wall time: 21.4 s

Using Dask::

    ### dask
    import os
    os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
    recnn.pd.set("modin")
    %%time
    env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

    100%|██████████| 138493/138493 [00:06<00:00, 19785.99it/s]
    CPU times: user 14.2 s, sys: 2.13 s, total: 16.3 s
    Wall time: 22 s
    <recnn.data.env.FrameEnv at 0x7f623fb30250>


**Free 2x improvement in loading speed**
