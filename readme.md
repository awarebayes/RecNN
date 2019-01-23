# RecNN


RecNN is reinforecement learning project for personalized news reccomendation written in pytorch. It follows [this paper](https://arxiv.org/pdf/1810.12027.pdf).

### Medium Articles
I wrote some medium articles explaining how this works:
  -  [Deep Reinforcement Learning for News Recommendation. Part 1: Architecture.](https://towardsdatascience.com/deep-reinforcement-learning-for-news-recommendation-part-1-architecture-5741b1a6ed56)
  -  (In Progress) Deep Reinforcement Learning for News Recommendation. Part 2: Simple implementation, DDPG with HER.
  -  (In Progress) Deep Reinforcement Learning for News Recommendation. Part 3: Scaling up. A3C and learning with multimpe GPUs.
  - (In Progress) Deep Reinforcement Learning for News Recommendation. Part 4: D4PG. Bayesian exploration.

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

