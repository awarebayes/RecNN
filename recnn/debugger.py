import numpy as np
import matplotlib.pyplot as plt
from . import plot
from scipy.special import kl_div
from scipy import stats
import torch

class Debugger:
    def __init__(self):
        self.debug_dict = {'error': {}, 'obj': {}, 'mat': {}}

    def log_error(self, name, x, test=False):
        if test:
            name = 'test ' + name
        try:
            target = self.debug_dict['error'][name]
        except KeyError:
            if name not in self.debug_dict['error']:
                self.debug_dict['error'][name] = {}
                target = self.debug_dict['error'][name]
                target['std'] = []
                target['mean'] = []
        target['std'].append(x.std().item())
        target['mean'].append(x.mean().item())

    def log_object(self, name, x, key='mat', test=False):
        if test:
            name = 'test ' + name
        self.debug_dict[key][name] = x

    def err_plot(self):
        for key, error in self.debug_dict['error'].items():
            sf = int(np.sqrt(len(error['mean'])))
            plt.errorbar(range(len(error['mean'][::sf])), error['mean'][::sf], error['std'][::sf])
            plt.title(key)
            plt.show()

    def matshow(self, key, range=slice(0, 50)):
        plot.pairwise_distances(self.debug_dict['mat'][key][range])

    @staticmethod
    def plot_kde_reconstruction_error(ad, gen_actions, gen_test_actions, true_actions, device=torch.device('cpu')):
        true_scores = ad.rec_error(torch.tensor(true_actions).to(device).float()).detach().cpu().numpy()
        gen_scores = ad.rec_error(torch.tensor(gen_actions).to(device).float()).detach().cpu().numpy()
        gen_test_scores = ad.rec_error(torch.tensor(gen_test_actions).to(device).float()).detach().cpu().numpy()

        true_kernel = stats.gaussian_kde(true_scores)
        gen_kernel = stats.gaussian_kde(gen_scores)
        gen_test_kernel = stats.gaussian_kde(gen_test_scores)

        x = np.linspace(0, 250, 100)
        probs_true = true_kernel(x)
        probs_gen = gen_kernel(x)
        probs_gen_test = gen_test_kernel(x)
        plt.figure(figsize=(16, 9))
        plt.plot(x, probs_true, '-b', label='true dist')
        plt.plot(x, probs_gen, '-r', label='generated dist')
        plt.plot(x, probs_gen_test, '-g', label='generated test dist')
        plt.legend()
        plt.show()

