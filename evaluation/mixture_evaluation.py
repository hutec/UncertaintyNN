from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from data import sample_generators

import numpy as np
import plotting

from training.mixture_training import mixture_training


def mixture_evaluation(x, y, dropout, learning_rate, epochs, n_mixtures, ax):
    sess, x_placeholder, dropout_placeholder = \
        mixture_training(x, y, dropout, learning_rate, epochs, n_mixtures)

    prediction_op = sess.graph.get_collection("prediction")
    uncertainty_op = sess.graph.get_collection("uncertainties")
    gmm_op = sess.graph.get_collection("gmm")

    additional_range = 0.2 * np.max(x)
    x_eval = np.linspace(np.min(x) - additional_range, np.max(x) + additional_range, 100).reshape([-1, 1])
    feed_dict = {x_placeholder: x_eval,
                 dropout_placeholder: 0}

    y_eval, uncertainties_eval = sess.run([prediction_op, uncertainty_op], feed_dict)
    y_eval = y_eval[0].flatten()

    aleatoric_eval, epistemic_eval = uncertainties_eval[0]
    total_uncertainty_eval = aleatoric_eval + epistemic_eval

    plotting.plot_mean_vs_truth_with_uncertainties(x, y, x_eval, y_eval, aleatoric_eval, epistemic_eval, ax)

    ax.legend()


if __name__ == "__main__":
    mixture_values = [1, 3, 5, 10]
    fig, axs = plt.subplots(len(mixture_values), 1, figsize=(30, 5*len(mixture_values)), sharey=True)
    fig.suptitle('Mixture-Model | Epochs: 20000, Learning Rate: 1e-3, Dropout 0.3', fontsize=20)
    x, y = sample_generators.generate_osband_sin_samples()
    for n_mixtures, ax in zip(mixture_values, axs):
        ax.set_title("%d Mixtures" % n_mixtures)
        mixture_evaluation(x, y, 0.3, 1e-3, n_mixtures=n_mixtures, epochs=15000, ax=ax)
        fig.savefig("Mixture_Sinus.pdf")

    fig, axs = plt.subplots(len(mixture_values), 1, figsize=(30, 5*len(mixture_values)), sharey=True)
    fig.suptitle('Mixture-Model | Epochs: 20000, Learning Rate: 1e-3, Dropout 0.3', fontsize=20)
    x, y = sample_generators.generate_osband_nonlinear_samples()
    for n_mixtures, ax in zip(mixture_values, axs):
        ax.set_title("%d Mixtures" % n_mixtures)
        mixture_evaluation(x, y, 0.3, 1e-3, n_mixtures=n_mixtures, epochs=15000, ax=ax)
        fig.savefig("Mixture_Nonlinear.pdf")



