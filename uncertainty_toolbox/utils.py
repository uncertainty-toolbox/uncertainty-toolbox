"""
Util functions for the toolbox.
"""

import numpy as np


def is_flat_same_shape(*args):
    if not isinstance(args[0], np.ndarray):
        return False
    first_shape = args[0].shape
    for arr in args:
        if not isinstance(arr, np.ndarray):
            return False
        if len(arr.shape) != 1:
            return False
        if arr.shape != first_shape:
            return False
    return True
