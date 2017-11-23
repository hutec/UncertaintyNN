import tensorflow as tf


def dropout_cnn_mnist_model(x, dropout_rate):
    """
    Builds a simple CNN MNIST classifier

    :param x:
    :return:
    """
    input_layer = tf.reshape(x, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(conv1, dropout_rate, training=True)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(conv2, dropout_rate, training=True)

    pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(dense, dropout_rate, training=True)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=10)
    class_prob = tf.nn.softmax(logits, name="softmax_tensor")

    return logits, class_prob


def combined_cnn_mnist_model(x, dropout_rate):
    pass

def bootstrap_cnn_mnist_model(x, n_heads):
    pass
