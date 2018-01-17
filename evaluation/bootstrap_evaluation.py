from data import sample_generators

import numpy as np
import plotting

from training.bootstrap_training import bootstrap_training
import matplotlib.pyplot as plt
# plt.switch_backend("Agg")
import seaborn as sns


def bootstrap_evaluation_depre(x, y, pred_range, n_heads=5, learning_rate=0.01, epochs=10000, display_step=2000):
    # Plotting

    # axs[0]: single heads
    # axs[1]: mean and std
    fig, axs = plt.subplots(2, 1, sharey=True, figsize=(20, 20))

    pred_x = pred_range.reshape([-1, 1])
    pred_ys = []
    for i, head in enumerate(heads):
        pred_y = sess.run(head, feed_dict={x_data: pred_x})
        pred_ys.append(pred_y)
        axs[0].plot(pred_x, pred_y, label="Head " + str(i))
    axs[0].scatter(x, y, label="training samples", alpha=0.3)

    y_squeezed = np.squeeze(pred_ys, axis=2).transpose()
    y_mean = np.mean(y_squeezed, axis=1)
    y_var = np.std(y_squeezed, axis=1)
    axs[0].legend()
    axs[0].fill_between(np.squeeze(pred_x, axis=1), y_mean - y_var, y_mean + y_var, alpha=0.3)

    lower = axs[0].get_ylim()[0]
    axs[0].fill_between(np.squeeze(pred_x, axis=1), lower, lower + y_var, alpha=0.1)

    _ = axs[0].set_title("Different head approximations and their Standard Deviation")

    #axs[1].fill_between(np.squeeze(pred_x, axis=1), 0, y_var, alpha=0.5)

    axs[1].scatter(x, y)
    axs[1].plot(pred_x, y_mean, linewidth=1, alpha=1)
    axs[1].fill_between(np.squeeze(pred_x, axis=1), y_mean - y_var, y_mean + y_var, alpha=0.1, color="blue")
    _ = axs[1].set_title("Mean and Standard Deviation")

    plt.suptitle("Bootstrap-Model | Heads: {}, Epochs: {}, Learning Rate {}".format(
        n_heads, epochs, learning_rate
    ))
    plt.show()
    return fig


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
