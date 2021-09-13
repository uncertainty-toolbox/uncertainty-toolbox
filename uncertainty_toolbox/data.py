"""
Code for importing and generating data.
"""
from typing import Tuple

import numpy as np


def synthetic_arange_random(
        num_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dataset of evenly spaced points and identity function (with some randomization).

    This returns predictions from some hypothetical uncertainty model.

    Args:
        num_points: The number of data points in the set.

    Returns:
        - The y predictions given by some synthetic model. These are the true
          values of y but with uniform noise added.
        - The standard deviation estimation given by some synthetic model. These
          are the errors between the predictions and the truth plus some unifom
          noise.
        - The true y labels.
        - The x data points.
    """
    x = np.arange(num_points)
    y_true = np.arange(num_points)
    y_pred = np.arange(num_points) + np.random.random((num_points,))
    y_std = np.abs(y_true - y_pred) + 0.1 * np.random.random((num_points,))

    return y_pred, y_std, y_true, x


def synthetic_sine_heteroscedastic(
        n_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return samples from "synthetic sine" heteroscedastic noisy function.

    This returns a synthetic dataset which one could use to train an uncertainty model.

    Args:
        n_points: The number of data points in the set.

    Returns:
        - The true y points of the dataset.
        - The standard deviation of the noise added.
        - The observed, noisy y data.
        - The x data points.
    """
    bounds = [0, 15]

    # x = np.random.uniform(bounds[0], bounds[1], n_points)
    x = np.linspace(bounds[0], bounds[1], n_points)

    f = np.sin(x)
    std = 0.01 + np.abs(x - 5.0) / 10.0
    noise = np.random.normal(scale=std)
    y = f + noise
    return f, std, y, x
