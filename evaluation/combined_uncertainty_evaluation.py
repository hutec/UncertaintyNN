from models import combined_uncertainty_model
from data import  sample_generators

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def combined_uncertainty_evaluation(x, y, pred_range, n_passes=50, dropout_rate=0.3, learning_rate=0.01, epochs=10000,
                                    display_step=2000):

    if 0 in x:
        x = np.delete(x, 0)
        y = np.delete(y, 0)

    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])

    x_data = tf.placeholder(tf.float32, [None, 1], name="x_data")
    y_data = tf.placeholder(tf.float32, [None, 1], name="y_data")

    predictions = combined_uncertainty_model.combined_model(x_data, dropout_rate)
    y_hat = tf.reshape(predictions[:, 0], [-1, 1])
    s = tf.reshape(predictions[:, 1], [-1, 1])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # TODO: Add weight decay
    combined_loss = tf.reduce_sum(0.5 * tf.exp(-1 * s) * tf.square(tf.abs(y_data - y_hat)) + 0.5 * s)

    train = optimizer.minimize(combined_loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    for epoch in range(epochs):
        sess.run(train, feed_dict={x_data: x, y_data: y})

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            loss = sess.run(combined_loss, feed_dict={x_data: x,
                                                      y_data: y})
            print("Loss: {}".format(loss))
            print("================")

    print("Training done")

    # Plotting
    pred_x = pred_range
    pred_x_multipass = np.array([[e] * n_passes for e in pred_x]).reshape([-1, 1])
    pred_x = pred_x.reshape([-1, 1])

    pred_y_multipass, s_multipass = sess.run([y_hat, s], feed_dict={x_data: pred_x_multipass})

    pred_y_multipass = pred_y_multipass.reshape(-1, n_passes)
    pred_y_mean = pred_y_multipass.mean(axis=1)
    pred_y_mean_squared = np.square(pred_y_multipass).mean(axis=1) # .reshape([-1, 1])
    pred_y_var = pred_y_multipass.var(axis=1)
    pred_y_std = pred_y_multipass.std(axis=1)

    # s = log(sigma^2)
    # ---> exp(s) = sigma^2
    sigma2_multipass = np.exp(s_multipass)
    sigma2_multipass = sigma2_multipass.reshape(-1, n_passes)
    sigma2_mean = sigma2_multipass.mean(axis=1)
    sigma2_var = sigma2_multipass.var(axis=1)
    sigma2_std = sigma2_multipass.std(axis=1)

    pred_epistemic_var = pred_y_var.reshape([-1, 1])

    combined_uncertainty = pred_y_mean_squared - np.square(pred_y_mean) + sigma2_mean

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(20, 20))
    ax.plot(pred_x, pred_y_mean, label="Predictive Mean ($\hat{y}$ mean of k passes)", linewidth=3)
    ax.plot(pred_x, pred_epistemic_var, label="Epistemic Variance (Variance of k passes)")
    ax.plot(pred_x, sigma2_mean, label="Aleatoric Variance ($\sigma^2$ mean of k passes)")
    ax.plot(pred_x, combined_uncertainty, label="Combined Uncertainty")
    ax.scatter(x, y, label="training samples", alpha=0.3)
    ax.set_ylim(np.min(y) - 0.3, np.max(y) + 0.3)

    ax.set_title("Combined Model | Passes: {}, Dropout: {}, Epochs: {}, Learning Rate: {}".format(
        n_passes, dropout_rate, epochs, learning_rate
    ))

    plt.legend()
    plt.show()
    sess.close()

    return fig


def combined_osband_sin_evaluation(n_samples=50, n_passes=50, dropout_rate=0.3, learning_rate=0.001, epochs=10000,
                                   display_step=2000):

    x, y = sample_generators.generate_osband_sin_samples(size=n_samples)
    pred_range = np.arange(-0.2, 1.2, 0.01)
    fig = combined_uncertainty_evaluation(x, y, pred_range, n_passes, dropout_rate, learning_rate, epochs, display_step)

    fig.savefig("results/combined_sinus_passes{}_dropout{}_samples{}_epochs{}_lr{}.pdf".format(
        n_passes, dropout_rate, n_samples, epochs, learning_rate
    ))


def combined_osband_nonlinear_evaluation(n_passes=50, dropout_rate=0.3, learning_rate=0.001, epochs=10000,
                                         display_step=2000):
    x, y = sample_generators.generate_osband_nonlinear_samples()
    pred_range = np.arange(-5, 5, 0.01)
    fig = combined_uncertainty_evaluation(x, y, pred_range, n_passes, dropout_rate, learning_rate, epochs, display_step)
    fig.savefig("results/combined_nonlinear_passes{}_dropout{}_epochs{}_lr{}.pdf".format(
        n_passes, dropout_rate, epochs, learning_rate
    ))


if __name__ == "__main__":
    combined_osband_sin_evaluation(epochs=100000)
    #combined_osband_nonlinear_evaluation(epochs=10000)
