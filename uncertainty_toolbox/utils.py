"""
Util functions for the toolbox.
"""

import sys
import numpy as np


def assert_is_flat_same_shape(*args):
    """Check if inputs are all same-length 1d numpy.ndarray."""

    try:
        assert isinstance(args[0], np.ndarray), "Error: all inputs must be of type numpy.ndarray"
        first_shape = args[0].shape
        for arr in args:
            assert isinstance(arr, np.ndarray), "Error: all inputs must be of type numpy.ndarray"
            assert len(arr.shape) == 1, "Error: all inputs must be 1d numpy.ndarray"
            assert arr.shape == first_shape, "Error: all inputs must have the same length"
    except AssertionError as error_msg:
        print(error_msg)
        sys.exit(1)

    return True
