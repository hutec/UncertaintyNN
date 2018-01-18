from models import dropout_model
from data import sample_generators

import tensorflow as tf


def dropout_training(x_truth, y_truth, dropout, learning_rate, epochs, display_step=2000):
    """
    Generic training of a Dropout Network for 2D data.

    :param x_truth: training samples x
    :param y_truth: training samples y / label
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return: session, x_placeholder, dropout_placeholder
    """
    tf.reset_default_graph()
    x_placeholder = tf.placeholder(tf.float32, [None, 1])
    y_placeholder = tf.placeholder(tf.float32, [None, 1])
    dropout_placeholder = tf.placeholder(tf.float32)

    prediction = dropout_model.dropout_model(x_placeholder, dropout_placeholder)

    tf.add_to_collection("prediction", prediction)

    loss = tf.losses.mean_squared_error(y_placeholder, prediction)

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

