"""
Code for importing and generating data.
"""

import numpy as np


def synthetic_arange_random(num_points=10):
    """
    Simple dataset of evenly spaced points and identity function (with some
    randomization)
    """
    y_true = np.arange(num_points)
    y_pred = np.arange(num_points) + np.random.random((num_points,))
    y_std = np.abs(y_true - y_pred) + .1 * np.random.random((num_points,))

    return (y_pred, y_std, y_true)


def synthetic_sine_heteroscedastic(n_points=10):
    """
    Return samples from "synthetic sine" heteroscedastic noisy function.
    """
    bounds = [0, 15]

    #x = np.random.uniform(bounds[0], bounds[1], n_points)
    x = np.linspace(bounds[0], bounds[1], n_points)

    f = np.sin(x)
    std = 0.01 + np.abs(x - 5.0) / 10.0
    noise = np.random.normal(scale=std)
    y = f + noise
    return f, std, y, x


def curvy_cosine(x):
    """
    Curvy cosine function.

    Parameters
    ----------
    x : ndarray
        2d numpy ndarray.
    """
    flat_neg_cos = np.sum(-1*np.cos(x), 1) / x.shape[1]
    curvy_cos = flat_neg_cos + 0.2 * np.linalg.norm(x, axis=1)
    curvy_cos = curvy_cos.reshape(-1, 1)
    return curvy_cos
