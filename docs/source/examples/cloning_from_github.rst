Cloning from github
===================

Pro tip: clone without history (unless you need it)::

    git clone --depth 1 git@github.com:awarebayes/RecNN.git 

Create ENV and install deps::

    conda create --name recnn
    conda activate recnn
    cd RecNN
    pip install -r requirements.txt

Download data from the donwloads section

Start jupyter notebook and jump to the examples folder ::

    jupyter-notebook .

Here is how my project directories looks like (shallow)::

    RecNN
    ├── .circleci
    ├── data
    ├── docs
    ├── examples
    ├── .git
    ├── .gitignore
    ├── LICENSE
    ├── models
    ├── readme.md
    ├── recnn
    ├── requirements.txt
    ├── res
    ├── runs
    ├── setup.cfg
    └── setup.py

Here is the data directory (ignore the cache)::

    data
    ├── cache
    │   ├── frame_env.pkl
    │   └── frame_env_truncated.pkl
    ├── embeddings
    │   └── ml20_pca128.pkl
    └── ml-20m
        ├── genome-scores.csv
        ├── genome-tags.csv
        ├── links.csv
        ├── movies.csv
        ├── ratings.csv
        ├── README.txt
        └── tags.csv