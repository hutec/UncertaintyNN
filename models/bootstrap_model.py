import tensorflow as tf


def bootstrap_model(x, dropout_rate, n_heads, mask):
    """
    Constructs model with n_heads bootstraps heads to process
    simple 2D data.

    :param x: input features x
    :param dropout_rate:
    :param n_heads: number of heads to use for bootstrapping
    :param mask: Mask to which heads are used for which samples

    :return: masked_heads, heads, mean, variance
    """
    keep_prob = 1 - dropout_rate

    with tf.variable_scope("bootstrap_heads"):
        heads = []
        for i in range(n_heads):
            fc1 = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)
            fc1 = tf.nn.dropout(fc1, keep_prob)
            fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu)
            heads.append(tf.layers.dense(inputs=fc2, units=1))

        heads = tf.stack(heads, axis=1)
        heads = mask_gradients(heads, mask)

    with tf.variable_scope("out"):
        mean, variance = tf.nn.moments(heads, axes=1)

    return heads, mean, variance


def mask_gradients(x, mask):
    """
    Helper function to propagate gradients only from masked heads.

    :param x: Tensor to be masked
    :param mask: Mask to select heads
    :return:
    """
    mask_h = tf.abs(mask - 1)
    return tf.stop_gradient(mask_h * x) + mask * x
