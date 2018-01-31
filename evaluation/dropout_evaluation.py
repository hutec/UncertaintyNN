from matplotlib.backends.backend_pdf import PdfPages
from data import sample_generators

import numpy as np
import plotting
import matplotlib.pyplot as plt

from training.dropout_training import dropout_training


def dropout_evaluation(x, y, dropout, learning_rate, epochs, n_passes, ax):
    sess, x_placeholder, dropout_placeholder = \
        dropout_training(x, y, dropout, learning_rate, epochs)

    prediction_op = sess.graph.get_collection("prediction")

    x_eval = np.linspace(1.1 * np.min(x), 1.1 * np.max(x), 100).reshape([-1, 1])

    feed_dict = {x_placeholder: x_eval,
                 dropout_placeholder: dropout}

    predictions = []
    for _ in range(n_passes):
        predictions.append(sess.run(prediction_op, feed_dict)[0])

    y_eval = np.mean(predictions, axis=0).flatten()
    uncertainty_eval = np.var(predictions, axis=0).flatten()

    plotting.plot_mean_vs_truth(x, y, x_eval, y_eval, uncertainty_eval, ax)


def dropout_osband_sin_evaluation(dropout, learning_rate, epochs, n_passes, ax=None):
    x, y = sample_generators.generate_osband_sin_samples()
    dropout_evaluation(x, y, dropout, learning_rate, epochs, n_passes, ax)


def dropout_osband_nonlinear_evaluation(dropout, learning_rate, epochs, n_passes, ax=None):
    x, y = sample_generators.generate_osband_nonlinear_samples()
    dropout_evaluation(x, y, dropout, learning_rate, epochs, n_passes, ax)


if __name__ == "__main__":
    dropout_values = [0.1, 0.3, 0.5, 0.7]
    fig, axs = plt.subplots(len(dropout_values), 1, figsize=(30, 5*len(dropout_values)))
    for dropout, ax in zip(dropout_values, axs):
        dropout_osband_sin_evaluation(dropout, 1e-3, 20000, 100, ax)
        fig.savefig("Dropout_Sinus.png")

    fig, axs = plt.subplots(len(dropout_values), 1, figsize=(30, 5*len(dropout_values)))
    for dropout, ax in zip(dropout_values, axs):
        dropout_osband_nonlinear_evaluation(dropout, 1e-3, 20000, 100, ax)
        fig.savefig("Dropout_Nonlinear.png")

