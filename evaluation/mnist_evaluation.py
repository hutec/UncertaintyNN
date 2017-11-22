from models import mnist_model
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def mnist_dropout_evaluation(n_passes=50, dropout_rate=0.3, learning_rate=1e-4, epochs=20000, display_step=2000):
    """

    :param n_passes:
    :param dropout_rate:
    :param learning_rate:
    :param epochs:
    :param display_step:
    :return:
    """
    mnist = input_data.read_data_sets("../data/MNIST-data", one_hot=True)

    x_data = tf.placeholder(tf.float32, shape=[None, 784])
    y_data = tf.placeholder(tf.float32, shape=[None, 10])
    dropout_rate_data = tf.placeholder(tf.float32)

    logits, class_prob = mnist_model.dropout_cnn_mnist_model(x_data, dropout_rate_data)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
    correct_prediction = tf.equal(tf.argmax(class_prob, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    for epoch in range(epochs):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x_data: batch[0], y_data: batch[1], dropout_rate_data: 0.5})

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))
            # cur_loss = sess.run(loss, feed_dict={x_data: batch[0],
            #                                      y_data: batch[1]})
            train_accuracy = sess.run(accuracy, feed_dict={
                x_data: batch[0], y_data: batch[1], dropout_rate_data:0})

            print("Accuracy: {}".format(train_accuracy))


if __name__ == "__main__":
    mnist_dropout_evaluation()
