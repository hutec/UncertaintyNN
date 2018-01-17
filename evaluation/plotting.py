from colors import theme

import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma


def plot_mean_vs_truth(x_truth, y_truth, x_prediction, y_prediction, std=None):
    """

    :param x_truth: x training sample
    :param y_truth: y training sample
    :param x_prediction: x evaluation
    :param y_prediction: y evaluation (predicted)
    :param std: (optional) standard deviation for every prediction
    :return: fig, ax
    """
    fig, ax = plt.subplots(1, 1, figsize=(30, 10))
    ax.scatter(x_truth, y_truth, label="Truth", color=theme["truth"])
    ax.plot(x_prediction, y_prediction, label="Prediction", color=theme["prediction_mean"])

    if std is not None:
        ax.fill_between(x_prediction.flatten(), y_prediction - std, y_prediction + std,
                        color=theme["prediction_std"], alpha=0.3, label="Prediction std")
    ax.legend()

    return fig, ax


def plot_mean_vs_truth_with_uncertainties(x_truth, y_truth, x_prediction, y_prediction,
                                          aleatoric, epistemic):
    """
    Same as plot_mean_vs_truth but with the uncertainties splitted into aleatoric and epistemic.

    :param x_truth:
    :param y_truth:
    :param x_prediction:
    :param y_prediction:
    :param std:
    :return: fig, ax
    """
    pass

def plot_gmm_weights(gmm_weights):
    """Plot GMM weights over samples"""


def plot_samples_per_head(x, y, n_heads, mask):
    """
    Plots the sample masks per head

    :param x:
    :param y:
    :param n_heads:
    :param mask:
    :return:
    """
    f, axs = plt.subplots(n_heads, 1, sharey=True, figsize=(10,20))
    for i in range(n_heads):
        ax = axs[i]
        masked_x = ma.masked_array(x, mask[i]).compressed().reshape([-1, 1])
        masked_y = ma.masked_array(y, mask[i]).compressed().reshape([-1, 1])
        ax.set_title(str(i))
        ax.scatter(masked_x, masked_y, label=str(i))
