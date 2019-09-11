from scipy.spatial import distance
from scipy import ndimage
import matplotlib.pyplot as plt
import torch
from scipy import stats
import numpy as np

def pairwise_distances_fig(embs):
    embs = embs.detach().cpu().numpy()
    similarity_matrix_cos = distance.cdist(embs, embs, 'cosine')
    similarity_matrix_euc = distance.cdist(embs, embs, 'euclidean')

    fig = plt.figure(figsize=(16,10))

    ax = fig.add_subplot(121)
    cax = ax.matshow(similarity_matrix_cos)
    fig.colorbar(cax)
    ax.set_title('Cosine')
    ax.axis('off')

    ax = fig.add_subplot(122)
    cax = ax.matshow(similarity_matrix_euc)
    fig.colorbar(cax)
    ax.set_title('Euclidian')
    ax.axis('off')

    fig.suptitle('Action pairwise distances')
    return fig


def pairwise_distances(embs):
    fig = pairwise_distances_fig(embs)
    fig.show()


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def smooth_gauss(arr, var):
    return ndimage.gaussian_filter1d(arr, var)


class Plotter:
    def __init__(self, debugger, style):
        self.debugger = debugger
        self.style = style
        self.smoothing = lambda x: smooth_gauss(x, 4)

    def set_smoothing_func(self, f):
        self.smoothing = f

    def plot_loss(self):
        for row in self.style:
            fig, axes = plt.subplots(1, len(row), figsize=(16, 6))
            if len(row) == 1: axes = [axes]
            for col in range(len(row)):
                key = row[col]
                axes[col].set_title(key)
                axes[col].plot(self.debugger.debug_dict['loss']['train']['step'],
                               self.smoothing(self.debugger.debug_dict['loss']['train'][key]), 'b-',
                               label='train')
                axes[col].plot(self.debugger.debug_dict['loss']['test']['step'],
                               self.debugger.debug_dict['loss']['test'][key], 'r-.',
                               label='test')
            plt.legend()
        plt.show()

    @staticmethod
    def kde_reconstruction_error(ad, gen_actions, gen_test_actions, true_actions, device=torch.device('cpu')):
        true_scores = ad.rec_error(torch.tensor(true_actions).to(device).float()).detach().cpu().numpy()
        gen_scores = ad.rec_error(torch.tensor(gen_actions).to(device).float()).detach().cpu().numpy()
        gen_test_scores = ad.rec_error(torch.tensor(gen_test_actions).to(device).float()).detach().cpu().numpy()

        true_kernel = stats.gaussian_kde(true_scores)
        gen_kernel = stats.gaussian_kde(gen_scores)
        gen_test_kernel = stats.gaussian_kde(gen_test_scores)

        x = np.linspace(0, 1000, 100)
        probs_true = true_kernel(x)
        probs_gen = gen_kernel(x)
        probs_gen_test = gen_test_kernel(x)

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        ax.plot(x, probs_true, '-b', label='true dist')
        ax.plot(x, probs_gen, '-r', label='generated dist')
        ax.plot(x, probs_gen_test, '-g', label='generated test dist')
        ax.legend()
        return fig

    @staticmethod
    def plot_kde_reconstruction_error(*args, **kwargs):
        fig = Plotter.kde_reconstruction_error(*args, **kwargs)
        fig.show()

