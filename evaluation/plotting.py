import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma


def plot_samples_per_head(x, y, n_heads, mask):
    """
    Plots the sample masks per head

    :param x:
    :param y:
    :param n_heads:
    :param mask:
    :return:
    """
    f, axs = plt.subplots(n_heads, 1, sharey=True, figsize=(10,20))
    for i in range(n_heads):
        ax = axs[i]
        masked_x = ma.masked_array(x, mask[i]).compressed().reshape([-1, 1])
        masked_y = ma.masked_array(y, mask[i]).compressed().reshape([-1, 1])
        ax.set_title(str(i))
        ax.scatter(masked_x, masked_y, label=str(i))
