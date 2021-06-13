import numpy as np


def check_flat_same_shape(*args):

    first_shape = args[0].shape
    for arr in args:
        try:
            type_condition = isinstance(arr, np.ndarray)
            dim_condition = len(arr.shape) == 1
            shape_condition = arr.shape == first_shape
        except:
            type_condition = False
            dim_condition = False
            shape_condition = False

        if not (type_condition and dim_condition and shape_condition):
            raise RuntimeError(
                "Input must be flat, 1D numpy arrays of same shape"
            )
