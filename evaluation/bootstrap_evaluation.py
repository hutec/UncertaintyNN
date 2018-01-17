from data import sample_generators

import numpy as np
import plotting

from training.bootstrap_training import bootstrap_training


# plt.suptitle("Bootstrap-Model | Heads: {}, Epochs: {}, Learning Rate {}".format(

def bootstrap_evaluation(x, y, dropout, learning_rate, epochs, n_heads):
    sess, x_placeholder, dropout_placeholder, mask_placeholder =\
        bootstrap_training(x, y, dropout, learning_rate, epochs, n_heads)

    prediction_op = sess.graph.get_collection("prediction")
    uncertainty_op = sess.graph.get_collection("uncertainties")

    x_eval = np.linspace(1.1 * np.min(x), 1.1 * np.max(x), 100).reshape([-1, 1])
    feed_dict = {x_placeholder: x_eval,
                 dropout_placeholder: 0,
                 mask_placeholder: np.ones(shape=(len(x_eval), n_heads, 1))}

    y_eval, uncertainties_eval = sess.run([prediction_op, uncertainty_op], feed_dict)
    y_eval = y_eval[0].flatten()
    uncertainties_eval = uncertainties_eval[0].flatten()

    fig, ax, = plotting.plot_mean_vs_truth(x, y,
                                           x_eval, y_eval, uncertainties_eval)

    return fig, ax
    # fig.savefig("results/bootstrap_sinus_heads{}_samples{}_epochs{}_lr{}.pdf".format(
    #     n_heads, n_samples, epochs, learning_rate
    # ))


def bootstrap_osband_sin_evaluation(dropout, learning_rate, epochs, n_heads):
    x, y = sample_generators.generate_osband_sin_samples()
    fig, ax = bootstrap_evaluation(x, y, dropout, learning_rate, epochs, n_heads)
    return fig, ax


def bootstrap_osband_nonlinear_evaluation(dropout, learning_rate, epochs, n_heads):
    x, y = sample_generators.generate_osband_nonlinear_samples()
    fig, ax = bootstrap_evaluation(x, y, dropout, learning_rate, epochs, n_heads)
    return fig, ax


if __name__ == "__main__":
    #bootstrap_osband_sin_evaluation(n_samples=50, n_heads=5, epochs=12000)
    f, a = bootstrap_osband_nonlinear_evaluation(0.3, 1e-3, n_heads=10, epochs=20000)
    from IPython import embed
    embed()
