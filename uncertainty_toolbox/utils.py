"""
Util functions for the toolbox.
"""
from typing import Any, NoReturn, Union

import numpy as np


def assert_is_flat_same_shape(*args: Any) -> Union[bool, NoReturn]:
    """Check if inputs are all same-length 1d numpy.ndarray.

    Args:
        args: the numpy arrays to check.

    Returns:
        True if all arrays are flat and the same shape, or else raises assertion error.
    """

    assert isinstance(args[0], np.ndarray), "All inputs must be of type numpy.ndarray"
    first_shape = args[0].shape
    for arr in args:
        assert isinstance(arr, np.ndarray), "All inputs must be of type numpy.ndarray"
        assert len(arr.shape) == 1, "All inputs must be 1d numpy.ndarray"
        assert arr.shape == first_shape, "All inputs must have the same length"

    return True


def assert_is_positive(*args: Any) -> Union[bool, NoReturn]:
    """Assert that all numpy arrays are positive.

    Args:
        args: the numpy arrays to check.

    Returns:
        True if all elements in all arrays are positive values, or else raises assertion error.
    """
    for arr in args:
        assert isinstance(arr, np.ndarray), "All inputs must be of type numpy.ndarray"
        assert all(arr > 0.0)

    return True
