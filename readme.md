<p align="center"> 
<img src="./res/logo.png">
</p>

This is my school project. It focuses on Reinforcement Learning, but there are many other things I learned during the development. Key topics: time series analysis, static dataset optimization, data preparation, and EDA. It also features my code for ML20 dataset that allows iterating through the dataset in a matter of 3 minutes. As well as my custom movie embeddings. DDPG doesn't seem to be working because it exploits the Value Network by recommending the same movie over and over again. But TD3 seems to be working just fine! You can see the distance matrices for the generated actions[below](#td3-results)


## Dataset Description
This project is built for MovieLens 20M dataset, but support for other datasets is in perspective. I have parsed all the movies in the '/links.csv' to get all auxiliary data from TMDB/IMDB. Text information was fed into Google's BERT/ OpenAI GPT2 models to get text embeddings. If you want to download anything, the links are down the description.

I also added static SARSA-like HDF5 dataset support so it takes ~3 minutes to get through all the ML20M dataset. Dynamically built it used to take about 2 hours but now you can iterate through 40GB of data in a matter of 3 minutes! You can generate static data yourself. I cannot upload it due to the legal reasons and slow internet.

Here is an overview:

- State - [None, frame_size * (embed_size+1) ] - PCA encoded previous actions (watched movies) embedding + rewards (ratings). All flattered and connected together
- Action - [None, embed_size] - PCA encoded current action embedding
- Reward - [None] - Integer, indicates whether the user liked the action or not
- Next state - look state - + Next state is basically the same but shifted +1 time step
- Done - [None] - Boolean, needed for TD(1)

## Misc Data

Everything of the misc sort is in the 'Misc Data' you can download in the downloads section, featuring all sorts of auxiliary stuff. Primarily it is movie info. If you don't want to use the embeddings, or just want to have some debug info/data for application this is what you need.

All text information is located in texts_bert.p / texts_gpt2.p in a dict {movie_id: numpy_array} format.

All of the categorical features had been label encoded, numerical standardized.

Here is an example of how the movie information looks like:

```python
{'adult': False,
 'collection': 210,
 'genres': [14, 1, 11],
 'original_language': 0,
 'popularity': 5.218749755002595,
 'production_companies': [96],
 'production_countries': [0],
 'release_year': 1995,
 'release_month': 10,
 'revenue': 4.893588591235185,
 'runtime': -0.5098445413830461,
 'spoken_languages': [0],
 'title': 'Toy Story',
 'vote_average': 1.2557064312220563,
 'vote_count': 1.8032194192281197,
 'budget': 1.1843770075921112,
 'revenue_d': 5.626649137875692}
```

## How to use static MovieLens Dataset in your project

```
import h5py

# include the file
f = h5py.File("*path to the static dataset*", "r")

# set some constants
batch = []
batch_size = 5000
n_batches = (f['state'].shape[0] // batch_size) + 1

def prepare_batch(*args):
    # device - torch.device cpu/cuda
    args = [torch.tensor(np.array(arg).astype(np.float)).to(device) for arg in args]
    return args

# iterate throught the batches
for i in range(n_batches):
    # get the batch
    batch = [f[key][i*batch_size:(i+1)*batch_size] for key in
             ['state', 'action', 'reward', 'next_state', 'done']]
    
    # do your framework-specific thing
    batch = prepare_batch(*batch)
	
    # do whatever you want here
	
    batch = []
```



### TD3 results
Here you can see the training process of the network:

<p align="center"> 
<img src="./res/Losses.png">
</p>

Here is a pairwise similarity matrix of real and generated actions (movie embeddings)

<p align="center"> 
<img src="./res/real_dist.png">
</p>
 

<p align="center"> 
<img src="./res/gen_dist.png">
</p>

It doesn't seem to overfit much. Here you can see the Kernel Density Estimation for Autoencoder Reconstruction scores. I use it as an anomaly detection metric. (Wasserstein Distance = ~50)

<p align="center"> 
<img src="./res/Anomaly_Detection.png">
</p>

 # Downloads
- [Movie Embeddings](https://drive.google.com/open?id=1kTyu05ZmtP2MA33J5hWdX8OyUYEDW4iI)
- [State Representation](https://drive.google.com/open?id=1DuNvPQ8pIxmZEFGNtXRSRxRcoWXU_0cO)
- [Misc Data](https://drive.google.com/open?id=1TclEmCnZN_Xkl3TfUXL5ivPYmLnIjQSu)

## FAQ:

 **What is Big and Small (Lite) dataset?**
 
For performance purposes, I added pre-trained state representation. It takes movie embeddings and ratings and encodes them into smaller 256 tensors. The Lite dataset utilizes this small trick whereas the big dataset does not.
 
**How to use state represrentation?**

 ```
 film_ids = [*watched films ids*]
 embeds = np.stack([movies_embeds[i] for i in film_ids])
 ratings = np.array([*ratings here*])
 state = state_rep(torch.tensor(np.concatenate([embeds, ratings])).float())
 ```
 
**What are the films ids?**
 
 It uses movies.csv from ML20M. The field is movieId
 
 **Something in the RL Losses looks weird**
 
It is fine for the RL losses. Keep in mind that RL algorithms utilize neural networks for calculating the loss functions (Policy) or some wacky stuff for Value.
 
 **What is the size of ...?**
 
| Name       | Dimensions Lite | Dimensions Big | Base Type |
|------------|-----------------|----------------|-----------|
| State      | 256             | 1290           | float     | 
| Action     | 128             | 128            | float     | 
| Reward     | 1               | 1              | int8      | 
| Next_State | 256             | 1290           | float     | 
| Done       | 1               | 1              | bool      | 

## Medium Articles (WIP)
I wrote some medium articles explaining how this works:
  -  [ Part 1: Architecture.](https://towardsdatascience.com/deep-reinforcement-learning-for-news-recommendation-part-1-architecture-5741b1a6ed56)
  - (In Progress) Part 2: Simple implementation DDPG. 
  - (In Progress) Part 3: TD3.

License
----

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

