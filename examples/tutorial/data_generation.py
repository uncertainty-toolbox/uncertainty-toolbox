"""
Generate synthetic data for the tutorial.
"""
import numpy as np

def create_data(num_data, is_uniform_noise=True):
    x = np.random.uniform(-10, 10, num_data)
    mu_y = mean_function(x)
    sigma_y = std_function(x)
    if is_uniform_noise:
        y = mean_function(x) + np.random.uniform(-np.sqrt(3) * sigma_y,
                                                 np.sqrt(3) * sigma_y)
    else:
        y = np.random.normal(loc=mean_function(x), scale=std_function(x))
    idxs = np.argsort(x)
    return x[idxs], y[idxs]

def mean_function(x):
    """Mean function for the labels given x."""
    return np.sin(x / 2) + x * np.cos(0.8 * x)

def std_function(x):
    """Give the standard deviation of the labels given x."""
    std = np.array(x)
    std[x < -5] = 1
    std[np.logical_and(x >= -5, x < 0)] = 0.01
    std[np.logical_and(x >= 0, x < 5)] = 1.5
    std[x >= 5] = 0.5
    return std
