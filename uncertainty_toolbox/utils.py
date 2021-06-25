"""
Util functions for the toolbox.
"""

import numpy as np


def assert_is_flat_same_shape(*args):
    assert isinstance(args[0], np.ndarray)
    first_shape = args[0].shape
    for arr in args:
        assert isinstance(arr, np.ndarray)
        assert len(arr.shape) == 1
        assert arr.shape == first_shape
    return True
