import tensorflow as tf
import tensorflow.contrib.layers as layers


def mixture_model(x, dropout_rate, n_mixtures):
    """
    Constructs a Mixture Density Network to process simple 2D data

    :param x: Input feature x
    :param dropout_rate:
    :param n_mixtures: Number of mixtures
    :return: gmm, mean, uncertainties
    """
    sigma_max = 5
    keep_prob = 1 - dropout_rate

    fc1 = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.layers.dense(inputs=fc1, units=50, activation=tf.nn.relu)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    with tf.variable_scope("gmm") as scope:
        # Gaussian Mixture Model (GMM) output used for loss
        output_layer = layers.fully_connected(fc2, n_mixtures * 3, scope=scope,
                                              activation_fn=None)
        raw_output = tf.reshape(output_layer, [-1, n_mixtures, 3])
        mixture_weights = tf.nn.softmax(raw_output[:, :, 0] - tf.expand_dims(tf.reduce_max(raw_output[:, :, 0],
                                                                                           axis=1), 1))
        mixture_means = raw_output[:, :, 1]
        mixture_variances = sigma_max * tf.sigmoid(raw_output[:, :, 2])

        # Stacking along axis=1 might be easier
        gmm = tf.stack([mixture_weights, mixture_means, mixture_variances])

    with tf.variable_scope("out"):
        # Mean and Uncertainties calculated on top of GMMs
        mean = tf.reduce_sum(mixture_weights * mixture_means, axis=1)
        aleatoric_uncertainty = tf.reduce_sum(mixture_weights * mixture_variances, axis=1)
        epistemic_uncertainty = tf.reduce_sum(mixture_weights *
                                              tf.square(mixture_means - tf.expand_dims(
                                                  tf.reduce_sum(mixture_weights * mixture_means, axis=1),
                                                  axis=1
                                              )), axis=1)

        uncertainties = tf.stack([aleatoric_uncertainty, epistemic_uncertainty])

    return gmm, mean, uncertainties

