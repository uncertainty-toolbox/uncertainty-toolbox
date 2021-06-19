"""
Code for importing and generating data.
"""

import numpy as np


def synthetic_arange_random(num_points=10):
    """
    Simple dataset of evenly spaced points and identity function (with some
    randomization).
    """
    x = np.arange(num_points)
    y_true = np.arange(num_points)
    y_pred = np.arange(num_points) + np.random.random((num_points,))
    y_std = np.abs(y_true - y_pred) + 0.1 * np.random.random((num_points,))

    return y_pred, y_std, y_true, x


def synthetic_sine_heteroscedastic(n_points=10):
    """
    Return samples from "synthetic sine" heteroscedastic noisy function.
    """
    bounds = [0, 15]

    # x = np.random.uniform(bounds[0], bounds[1], n_points)
    x = np.linspace(bounds[0], bounds[1], n_points)

    f = np.sin(x)
    std = 0.01 + np.abs(x - 5.0) / 10.0
    noise = np.random.normal(scale=std)
    y = f + noise
    return f, std, y, x
