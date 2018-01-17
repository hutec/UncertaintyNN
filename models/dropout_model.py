import tensorflow as tf


def dropout_model(x, dropout_rate):
    """
    Constructs Dropout network to process simple 2D data.
    After every weight layer a dropout layer is placed.

    :param x: Input feature x
    :param dropout_rate:
    :return: prediction
    """
    keep_prob = 1 - dropout_rate

    fc1 = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    output_layer = tf.layers.dense(inputs=fc2, units=1)
    predictions = tf.reshape(output_layer, [-1, 1])

    return predictions


