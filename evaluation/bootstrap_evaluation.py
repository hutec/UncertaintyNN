from matplotlib.backends.backend_pdf import PdfPages

from data import sample_generators

import matplotlib.pyplot as plt

import numpy as np
import plotting

from training.bootstrap_training import bootstrap_training


def bootstrap_evaluation(x, y, dropout, learning_rate, epochs, n_heads, ax):
    sess, x_placeholder, dropout_placeholder, mask_placeholder =\
        bootstrap_training(x, y, dropout, learning_rate, epochs, n_heads)

    prediction_op = sess.graph.get_collection("prediction")
    uncertainty_op = sess.graph.get_collection("uncertainties")
    heads_op = sess.graph.get_collection("heads")

    additional_range = 0.2 * np.max(x)
    x_eval = np.linspace(np.min(x) - additional_range, np.max(x) + additional_range, 100).reshape([-1, 1])
    feed_dict = {x_placeholder: x_eval,
                 dropout_placeholder: 0,
                 mask_placeholder: np.ones(shape=(len(x_eval), n_heads, 1))}

    y_eval, uncertainties_eval, heads_eval = sess.run([prediction_op, uncertainty_op, heads_op], feed_dict)
    heads_eval = np.array(heads_eval).reshape(len(x_eval), n_heads)
    y_eval = y_eval[0].flatten()
    uncertainties_eval = uncertainties_eval[0].flatten()

    for i in range(n_heads):
        ax.plot(x_eval, heads_eval[:, i], alpha=0.3)

    plotting.plot_mean_vs_truth(x, y, x_eval, y_eval, uncertainties_eval, ax)


if __name__ == "__main__":
    heads = [3, 5, 10, 15]
    fig, axs = plt.subplots(len(heads), 1, figsize=(30, 5*len(heads)), sharey=True)
    fig.suptitle('Bootstrap-Model | Epochs: 15000, Learning Rate: 1e-3', fontsize=20)
    x, y = sample_generators.generate_osband_sin_samples()
    for n_heads, ax in zip(heads, axs):
        ax.set_title("%d Heads" % n_heads)
        bootstrap_evaluation(x, y, 0.3, 1e-3, 15000, n_heads, ax)
        fig.savefig("Bootstrap_Sinus.pdf")

    fig, axs = plt.subplots(len(heads), 1, figsize=(30, 5*len(heads)), sharey=True)
    fig.suptitle('Bootstrap-Model | Epochs: 15000, Learning Rate: 1e-3', fontsize=20)
    x, y = sample_generators.generate_osband_nonlinear_samples()
    for n_heads, ax in zip(heads, axs):
        ax.set_title("%d Heads" % n_heads)
        bootstrap_evaluation(x, y, 0.2, 1e-3, 15000, n_heads, ax)
        fig.savefig("Bootstrap_Nonlinear.pdf")
