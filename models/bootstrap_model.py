import tensorflow as tf


def boostrap_model(x, n_heads=5):
    """
    Constructs model with n_heads bootstraps heads to process
    simple 2d data.

    :param x: input features
    :param n_heads: number of heads to use for bootstrapping

    :return: k_heads predictions
    """

    with tf.device("/gpu:0"):
        #x = tf.reshape(x, [-1, 1])
        fc1 = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)

        # Place second layer in shared network
        #fc2 = tf.layers.dense(inputs=fc1, units=20, activation=tf.nn.relu)

        heads = []
        for i in range(n_heads):
            fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu)
            heads.append(tf.layers.dense(inputs=fc2, units=1))

        return heads


