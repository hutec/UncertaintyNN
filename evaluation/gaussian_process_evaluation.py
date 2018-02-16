from matplotlib.backends.backend_pdf import PdfPages
from data import sample_generators

import numpy as np
import plotting
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

if __name__ == "__main__":
    x, y = sample_generators.generate_osband_sin_samples()
    additional_range = 0.1 * np.max(x)
    x_eval = np.linspace(np.min(x) - additional_range, np.max(x) + additional_range, 100).reshape([-1, 1])
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    fig, axs = plt.subplots(2, 1, figsize=(30, 10))
    kernel = 0.5 * RBF(length_scale=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.03, n_restarts_optimizer=10)

    y_prior = gp.sample_y(x_eval, 5)
    axs[0].plot(x_eval, y_prior)
    gp.fit(x, y)

    y_eval, sigma = gp.predict(x_eval, return_std=True)
    y_eval = y_eval.flatten()

    plotting.plot_mean_vs_truth(x, y, x_eval, y_eval, sigma, axs[1])
    axs[1].set_title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
                     % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.show()
    fig.savefig("GP_Sinus.pdf")
    plt.close()

    x, y = sample_generators.generate_osband_nonlinear_samples()
    additional_range = 0.2 * np.max(x)
    x_eval = np.linspace(np.min(x) - additional_range, np.max(x) + additional_range, 100).reshape([-1, 1])
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    fig, axs = plt.subplots(2, 1, figsize=(30, 10))
    kernel = 1 * RBF(length_scale=1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.5, n_restarts_optimizer=10)

    y_prior = gp.sample_y(x_eval, 5)
    axs[0].plot(x_eval, y_prior)
    gp.fit(x, y)

    y_eval, sigma = gp.predict(x_eval, return_std=True)
    y_eval = y_eval.flatten()

    plotting.plot_mean_vs_truth(x, y, x_eval, y_eval, sigma, axs[1])
    axs[1].set_title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
                     % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.show()
    fig.savefig("GP_Nonlinear.pdf")
    plt.close()

