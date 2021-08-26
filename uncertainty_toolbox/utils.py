"""
Util functions for the toolbox.
"""

import numpy as np


def assert_is_flat_same_shape(*args):
    """Check if inputs are all same-length 1d numpy.ndarray."""

    assert isinstance(
        args[0], np.ndarray
    ), "All inputs must be of type numpy.ndarray"
    first_shape = args[0].shape
    for arr in args:
        assert isinstance(
            arr, np.ndarray
        ), "All inputs must be of type numpy.ndarray"
        assert len(arr.shape) == 1, "All inputs must be 1d numpy.ndarray"
        assert arr.shape == first_shape, "All inputs must have the same length"

    return True


def assert_is_positive(*args):
    for arr in args:
        assert isinstance(
            arr, np.ndarray
        ), "All inputs must be of type numpy.ndarray"
        assert all(arr > 0.0)

    return True
