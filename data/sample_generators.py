import numpy as np
import pandas as pd


def generate_osband_sin_samples(size=30, variance_1=0.03, variance_2=0.03):
    """
    Generate samples from the function in the Osband paper.
    It allows to have different variances for the two intervals.

    :param size: number of samples
    :param variance_1: variance for x in [0, 0.6]
    :param variance_2: variance for x in [0.8, 1]
    :return: x, y
    """

    alpha = 4
    beta = 13

    w1 = np.random.normal(0, variance_1, size=int(0.75*size))
    x1 = np.linspace(0, 0.6, 0.75 * size)

    w2 = np.random.normal(0, variance_2, size=int(0.25*size))
    x2 = np.linspace(0.8, 1.0, 0.25 * size)

    x = np.append(x1, x2)
    w = np.append(w1, w2)

    y = x + np.sin(alpha * (x + w)) + np.sin(beta * (x + w)) + w

    return x, y


def generate_osband_nonlinear_samples():
    """
    Generates non-linear examples from Osband.

    :return: x, y
    """
    x = np.array([-1, -1, 0.01, 0.01, 1, 1])
    y = np.array([-1, 1, -1, 1, -1, 1])

    return x, y


def generate_linear_samples(size=30):
    """
    Generate simple linear samples

    :param size: number of samples
    :return: x, y
    """

    # TODO
    x = np.array(range(1000000))
    y = np.array(range(1000000))
    return x, y



