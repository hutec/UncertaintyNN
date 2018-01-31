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

    x_eval = np.linspace(1.1 * np.min(x), 1.1 * np.max(x), 100).reshape([-1, 1])
    feed_dict = {x_placeholder: x_eval,
                 dropout_placeholder: 0}

    y_eval, uncertainties_eval = sess.run([prediction_op, uncertainty_op], feed_dict)
    y_eval = y_eval[0].flatten()

    aleatoric_eval, epistemic_eval = uncertainties_eval[0]
    total_uncertainty_eval = aleatoric_eval + epistemic_eval

    plotting.plot_mean_vs_truth(x, y, x_eval, y_eval, total_uncertainty_eval, ax)


def mixture_osband_sin_evaluation(dropout, learning_rate, epochs, n_mixtures, ax=None):
    x, y = sample_generators.generate_osband_sin_samples()
    mixture_evaluation(x, y, dropout, learning_rate, epochs, n_mixtures, ax)


def mixture_osband_nonlinear_evaluation(dropout, learning_rate, epochs, n_mixtures, ax=None):
    x, y = sample_generators.generate_osband_nonlinear_samples()
    mixture_evaluation(x, y, dropout, learning_rate, epochs, n_mixtures, ax)


if __name__ == "__main__":
    mixture_values = [1, 3, 5, 10]
    fig, axs = plt.subplots(len(mixture_values), 1, figsize=(30, 5*len(mixture_values)))
    for n_mixtures, ax in zip(mixture_values, axs):
        mixture_osband_sin_evaluation(0.3, 1e-4, n_mixtures=n_mixtures, epochs=20000, ax=ax)
        fig.savefig("Mixture_Sinus.png")

    for n_mixtures, ax in zip(mixture_values, axs):
        mixture_osband_nonlinear_evaluation(0.3, 1e-4, n_mixtures=n_mixtures, epochs=20000, ax=ax)
        fig.savefig("Mixture_Nonlinear.png")



