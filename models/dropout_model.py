import tensorflow as tf


def dropout_model(x, dropout_rate=0.3):
    """
    Constructs dropout network with tf.layers to process
    simple 2d data.

    :param x: Input feature x
    :param dropout_rate: Dropout rate used during training and inference

    :return: Dropout network model predictions tensor
    """

    #with tf.device("/gpu:0"):

        # input_layer = tf.feature_column.input_layer(features, tf.feature_column.numeric_column("x"))
        #input_layer = tf.reshape(x, [-1, 1])

    fc1 = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)
    fc1 = tf.layers.dropout(fc1, dropout_rate, training=True)

    fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu)
    fc2 = tf.layers.dropout(fc2, dropout_rate, training=True)

    output_layer = tf.layers.dense(inputs=fc2, units=1)
    predictions = tf.reshape(output_layer, [-1, 1])

    return predictions


