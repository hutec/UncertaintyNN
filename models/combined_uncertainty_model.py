import tensorflow as tf


def combined_model(x, dropout_rate=0.3):
    """
    Model that combines aleatoric and epistemic uncertainty.
    Based on the "What uncertainties do we need" paper by Kendall.
    Works for simple 2d data.

    :param x:
    :param dropout_rate:
    :return:
    """

    # with tf.device("/gpu:0"):

    fc1 = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)
    fc1 = tf.layers.dropout(fc1, dropout_rate, training=True)

    fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu)
    fc2 = tf.layers.dropout(fc2, dropout_rate, training=True)

    # Output layers has predictive mean and variance sigma^2
    output_layer = tf.layers.dense(fc2, units=2)
    predictions = tf.reshape(output_layer, [-1, 2])

    return predictions
