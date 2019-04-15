# RecNN


RecNN is reinforecement learning project for personalized news reccomendation written in pytorch. It follows [this paper](https://arxiv.org/pdf/1810.12027.pdf).

This project is built for MovieLens 20M dataset, but support for other datasets is in perspective.
I have parsed all the movies in the '/links.csv' to get all auxiliary data. Text information was fed into Google's BERT/ OpenAI GPT2 models to get text embeddings. All the data can be found [here](https://drive.google.com/file/d/1TclEmCnZN_Xkl3TfUXL5ivPYmLnIjQSu/view?usp=sharing)

All text information is located in `texts_bert.p / texts_gpt2.p` in dict {movie_id: numpy_array} format.

All of cat features had been label encoded, numerical standardized.

Note: data is not frequently updated, but the notebook is designed to run each chapter independently. For instance, today I didn't add PCA movie embeddings, but you still can easily generate them yourself in a matter of 5 minutes or so.

Here is an example of how the movie info looks like:

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

### Medium Articles (Deep Reinforcement Learning for News Recommendation)
I wrote some medium articles explaining how this works:
  -  [ Part 1: Architecture.](https://towardsdatascience.com/deep-reinforcement-learning-for-news-recommendation-part-1-architecture-5741b1a6ed56)
  -  Part 2: Simple implementation DDPG. 
  - (In Progress) Part 3: D4PG. 
  - (In Progress) Part 4: Rainbow and ODEs. 

### Key Notes
  - Written in Pytorch 1.0
  - Multiprocessing GPU support. Asyncronous Learning.
  - DDPG with D4PG support.
  - A3C will only work in UNIX. Posix (Windows) is not supported for async learning. (VirtualBox and Docker will not help)
  - Notebooks may be found /notes
  - Ready-to-ship project will be located in /build

### Help and Commiting:   
> I will be glad to any cooperation and commiting to the project. We have a discord server. 

### Installation
```sh
$ git clone https://github.com/awarebayes/RecNN
$ cd RecNN
$ pip install requirements.txt
```

### Todo (Not implemented)

- Write Articles as I implement this
- Train news embeddings
- DDPG with HER
- A3C and async learning
- D4PG
- Docker Support

License
----

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

