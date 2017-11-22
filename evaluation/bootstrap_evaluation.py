from models import bootstrap_model
from data import  sample_generators

import tensorflow as tf
import numpy as np
from scipy.stats import bernoulli
import numpy.ma as ma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def bootstrap_evaluation(x, y, pred_range, n_heads=5, learning_rate=0.01, epochs=10000, display_step=2000):
    """
    Bootstrap network evaluation given x and y.

    :param x:
    :param y:
    :param pred_range:
    :param n_heads:
    :param epochs:
    :return:
    """
    assert len(x) == len(y)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    rv = bernoulli(0.5)
    mask = rv.rvs(size=(n_heads, len(x)))

    x_data = tf.placeholder(tf.float32, [None, 1])
    y_data = tf.placeholder(tf.float32, [None, 1])

    heads = bootstrap_model.boostrap_model(x_data, n_heads)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    loss_per_head = []
    train_per_head = []
    for head in heads:
        loss = tf.losses.mean_squared_error(y_data, head)
        loss_per_head.append(loss)
        train_per_head.append(optimizer.minimize(loss))

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    for epoch in range(epochs):
        for i, t in enumerate(train_per_head):
            masked_x = ma.masked_array(x, mask[i]).compressed().reshape([-1, 1])
            masked_y = ma.masked_array(y, mask[i]).compressed().reshape([-1, 1])
            sess.run(t, feed_dict={x_data: masked_x, y_data: masked_y})

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            for i, loss in enumerate(loss_per_head):
                curLoss = sess.run(loss, feed_dict={x_data: x.reshape([-1, 1]),
                                                    y_data: y.reshape([-1, 1])})
                print("Head {}: Loss {}".format(i, curLoss))
            print("================")

    print("Training done")


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


def bootstrap_osband_sin_evaluation(n_samples=50, n_heads=5, learning_rate=0.001, epochs=8000,
                                    display_step=2000):

    x, y = sample_generators.generate_osband_sin_samples(size=n_samples)
    pred_range = np.arange(-0.2, 1.2, 0.01)
    fig = bootstrap_evaluation(x, y, pred_range, n_heads, learning_rate, epochs, display_step)

    fig.savefig("results/bootstrap_sinus_heads{}_samples{}_epochs{}_lr{}.pdf".format(
        n_heads, n_samples, epochs, learning_rate
    ))


def bootstrap_osband_nonlinear_evaluation(n_heads=5, learning_rate=0.001, epochs=8000,
                                          display_step=2000):
    x, y = sample_generators.generate_osband_nonlinear_samples()
    pred_range = np.arange(-2.5, 2.5, 0.01)
    fig = bootstrap_evaluation(x, y, pred_range, n_heads, learning_rate, epochs, display_step)
    fig.savefig("results/bootstrap_nonlinear_heads{}_epochs{}_lr{}.pdf".format(
        n_heads, epochs, learning_rate
    ))


if __name__ == "__main__":
    bootstrap_osband_sin_evaluation(n_samples=50, n_heads=5, epochs=12000)
    bootstrap_osband_nonlinear_evaluation(n_heads=10, epochs=8000)
