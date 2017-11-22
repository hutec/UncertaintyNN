from models import dropout_model
from data import  sample_generators

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def dropout_evaluation(x, y, pred_range, n_passes=50, dropout_rate=0.3, learning_rate=0.001, epochs=20000, display_step=2000):

    """
    Generic dropout model evaluation
    :param x:
    :param y:
    :param n_passes:
    :param dropout_rate:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return:
    """
    x_data = tf.placeholder(tf.float32, [None, 1])
    y_data = tf.placeholder(tf.float32, [None, 1])

    predictions = dropout_model.dropout_model(x_data, dropout_rate)

    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])

    loss = tf.losses.mean_squared_error(y_data, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    for epoch in range(epochs):
        sess.run(train, feed_dict={x_data: x,
                                   y_data: y})

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            cur_loss = sess.run(loss, feed_dict={x_data: x,
                                                 y_data: y})
            print("Loss: {}".format(cur_loss))
    print("Training done.")

    pred_x = pred_range
    pred_x_multipass = np.array([[e] * n_passes for e in pred_x]).flatten()
    pred_x_multipass = pred_x_multipass.reshape([-1, 1])

    pred_y_multipass = sess.run(predictions, feed_dict={x_data: pred_x_multipass})
    pred_y_multipass = pred_y_multipass.reshape(-1, n_passes)
    pred_y_mean = pred_y_multipass.mean(axis=1)
    pred_y_var = pred_y_multipass.var(axis=1)
    pred_y_std = pred_y_multipass.std(axis=1)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(20, 20))
    ax.scatter(x, y, label="training samples", alpha=0.3)
    ax.plot(pred_x, pred_y_mean, color="red", label="prediction")
    ax.fill_between(pred_x, pred_y_mean - pred_y_var, pred_y_mean + pred_y_var, alpha=0.5)
    ax.fill_between(pred_x, 0, pred_y_var, alpha=0.5, label="variance")
    ax.legend()

    ax.set_title("Dropout-Model | Dropout: {}, Passes: {}, Epochs: {}, Learning Rate: {}".format(
        dropout_rate, n_passes, epochs, learning_rate
    ))
    plt.show()

    return fig


def dropout_osband_sin_evaluation(n_samples=50, n_passes=50, dropout_rate=0.4, learning_rate=0.001, epochs=20000,
                                  display_step=2000):
    x, y = sample_generators.generate_osband_sin_samples(size=n_samples)
    pred_range = np.arange(-0.2, 1.2, 0.01)
    fig = dropout_evaluation(x, y, pred_range, n_passes, dropout_rate, learning_rate, epochs, display_step)

    fig.savefig("results/dropout_sinus_passes{}_dropout{}_samples{}_epochs{}_lr{}.pdf".format(
        n_passes, dropout_rate, n_samples, epochs, learning_rate
    ))


def dropout_osband_nonlinear_evaluation(n_passes=50, dropout_rate=0.4, learning_rate=0.001, epochs=6000,
                                        display_step=2000):
    x, y = sample_generators.generate_osband_nonlinear_samples()
    pred_range = np.arange(-5, 5, 0.01)
    fig = dropout_evaluation(x, y, pred_range, n_passes, dropout_rate, learning_rate, epochs, display_step)
    fig.savefig("results/dropout_nonlinear_passes{}_dropout{}_epochs{}_lr{}.pdf".format(
        n_passes, dropout_rate, epochs, learning_rate
    ))


if __name__ == "__main__":
    dropout_osband_sin_evaluation()
    dropout_osband_nonlinear_evaluation()
