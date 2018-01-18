import tensorflow as tf


def combined_model(x, dropout_rate):
    """
    Model that combines aleatoric and epistemic uncertainty.
    Based on the "What uncertainties do we need" paper by Kendall.
    Works for simple 2D data.

    :param x: Input feature x
    :param dropout_rate:
    :return: prediction, log(sigma^2)
    """

    # with tf.device("/gpu:0"):
    keep_prob = 1 - dropout_rate

    fc1 = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Output layers has predictive mean and variance sigma^2
    output_layer = tf.layers.dense(fc2, units=2)

    predictions = tf.expand_dims(output_layer[:, 0], -1)
    log_variance = tf.expand_dims(output_layer[:, 1], -1)

    return predictions, log_variance
