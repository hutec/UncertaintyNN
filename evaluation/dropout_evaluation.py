from matplotlib.backends.backend_pdf import PdfPages
from data import sample_generators

import numpy as np
import plotting
import matplotlib.pyplot as plt

from training.dropout_training import dropout_training


def dropout_evaluation(x, y, dropout, learning_rate, epochs, n_passes, ax):
    # Hardcoded training dropout
    sess, x_placeholder, dropout_placeholder = \
        dropout_training(x, y, 0.2, learning_rate, epochs)

    prediction_op = sess.graph.get_collection("prediction")

    additional_range = 0.1 * np.max(x)
    x_eval = np.linspace(np.min(x) - additional_range, np.max(x) + additional_range, 100).reshape([-1, 1])

    feed_dict = {x_placeholder: x_eval,
                 dropout_placeholder: dropout}

    predictions = []
    for _ in range(n_passes):
        predictions.append(sess.run(prediction_op, feed_dict)[0])

    y_eval = np.mean(predictions, axis=0).flatten()
    uncertainty_eval = np.var(predictions, axis=0).flatten()

    plotting.plot_mean_vs_truth(x, y, x_eval, y_eval, uncertainty_eval, ax)


if __name__ == "__main__":
    dropout_values = [0.1, 0.3, 0.5, 0.6]
    fig, axs = plt.subplots(len(dropout_values), 1, figsize=(30, 5*len(dropout_values)), sharey=True)
    x, y = sample_generators.generate_osband_sin_samples()
    for dropout, ax in zip(dropout_values, axs):
        ax.set_title("%.3f Dropout" % dropout)
        dropout_evaluation(x, y, dropout, 1e-3, 20000, 100, ax)
        fig.savefig("Dropout_Sinus.pdf")

    fig, axs = plt.subplots(len(dropout_values), 1, figsize=(30, 5*len(dropout_values)), sharey=True)
    x, y = sample_generators.generate_osband_nonlinear_samples()
    for dropout, ax in zip(dropout_values, axs):
        ax.set_title("%.3f Dropout" % dropout)
        dropout_evaluation(x, y, dropout, 1e-3, 20000, 100, ax)
        fig.savefig("Dropout_Nonlinear.pdf")

