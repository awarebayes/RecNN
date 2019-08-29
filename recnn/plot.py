from scipy.spatial import distance
from scipy import ndimage
import matplotlib.pyplot as plt


def pairwise_distances(embs):
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
    def __init__(self):
        assert NotImplementedError
