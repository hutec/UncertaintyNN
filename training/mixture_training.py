from models import mixture_model
from data import sample_generators

import tensorflow as tf


def mixture_training(x_truth, y_truth, dropout, learning_rate, epochs, n_mixtures, display_step=2000):
    """
    Generic training of a Mixture Density Mixture Network for 2D data.

    :param x_truth: training samples x
    :param y_truth: training samples y / label
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param n_mixtures: Number of mixtures in GMM
    :param display_step:
    :return: session, x_placeholder, dropout_placeholder
    """
    tf.reset_default_graph()
    x_placeholder = tf.placeholder(tf.float32, [None, 1])
    y_placeholder = tf.placeholder(tf.float32, [None, 1])
    dropout_placeholder = tf.placeholder(tf.float32)
    eps = 1e-4

    gmm, mean, uncertainties = mixture_model.mixture_model(x_placeholder, dropout_placeholder, n_mixtures=n_mixtures)

    tf.add_to_collection("gmm", gmm)
    tf.add_to_collection("prediction", mean)
    tf.add_to_collection("uncertainties", uncertainties)

    mixture_weights = gmm[0]
    mixture_means = gmm[1]
    mixture_variances = gmm[2]

    dist = tf.distributions.Normal(loc=mixture_means, scale=mixture_variances)
    loss = - tf.reduce_mean(
        tf.log(tf.reduce_sum(mixture_weights * dist.prob(y_placeholder), axis=1) + eps),
        axis=0
    )

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    for epoch in range(epochs):
        feed_dict = {x_placeholder: x_truth.reshape([-1, 1]),
                     y_placeholder: y_truth.reshape([-1, 1]),
                     dropout_placeholder: dropout}

        sess.run(train, feed_dict=feed_dict)

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            current_loss = sess.run(loss, feed_dict=feed_dict)
            print("Loss {}".format(current_loss))
            print("================")

    print("Training done")

    return sess, x_placeholder, dropout_placeholder


if __name__ == "__main__":
    x, y = sample_generators.generate_osband_nonlinear_samples()
    mixture_training(x, y, 0.3, 1e-3, 20000)




