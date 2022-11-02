"""
Util functions for the toolbox.
"""
from typing import Any, NoReturn, Tuple, Union

from numbers import Number
import numpy as np
import torch

Numeric = Union[int, float, np.ndarray]


def assert_is_flat_same_shape(*args: Any) -> Union[bool, NoReturn]:
    """Check if inputs are all same-length 1d numpy.ndarray.

    Args:
        args: the numpy arrays to check.

    Returns:
        True if all arrays are flat and the same shape, or else raises assertion error.
    """
    assert len(args) > 0
    assert isinstance(args[0], np.ndarray) or isinstance(args[0], torch.Tensor), "All inputs must be of type numpy.ndarray or torch.Tensor"
    first_shape = args[0].shape
    for arr in args:
        assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor), "All inputs must be of type numpy.ndarray or torch.Tensor"
        assert len(arr.shape) == 1, "All inputs must be 1d numpy.ndarray or torch.Tensor"
        assert arr.shape == first_shape, "All inputs must have the same length"

    return True


def assert_is_positive(*args: Any) -> Union[bool, NoReturn]:
    """Assert that all numpy arrays are positive.

    Args:
        args: the numpy arrays to check.

    Returns:
        True if all elements in all arrays are positive values, or else raises assertion error.
    """
    assert len(args) > 0
    for arr in args:
        assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor), "All inputs must be of type numpy.ndarray or torch.Tensor"
        if isinstance(arr, np.ndarray):
            assert np.all(arr > 0.0)
        else:
            assert (arr > 0.0).all()

    return True





def trapezoid_area(
    xl: np.ndarray,
    al: np.ndarray,
    bl: np.ndarray,
    xr: np.ndarray,
    ar: np.ndarray,
    br: np.ndarray,
    absolute: bool = True,
) -> Numeric:
    """
    Calculate the area of a vertical-sided trapezoid, formed connecting the following points:
        (xl, al) - (xl, bl) - (xr, br) - (xr, ar) - (xl, al)

    This function considers the case that the edges of the trapezoid might cross,
    and explicitly accounts for this.

    Args:
        xl: The x coordinate of the left-hand points of the trapezoid
        al: The y coordinate of the first left-hand point of the trapezoid
        bl: The y coordinate of the second left-hand point of the trapezoid
        xr: The x coordinate of the right-hand points of the trapezoid
        ar: The y coordinate of the first right-hand point of the trapezoid
        br: The y coordinate of the second right-hand point of the trapezoid
        absolute: Whether to calculate the absolute area, or allow a negative area (e.g. if a and b are swapped)

    Returns: The area of the given trapezoid.

    """

    # Differences
    dl = bl - al
    dr = br - ar

    # The ordering is the same for both iff they do not cross.
    cross = dl * dr < 0

    # Treat the degenerate case as a trapezoid
    cross = cross * (1 - ((dl == 0) * (dr == 0)))

    # trapezoid for non-crossing lines
    area_trapezoid = (xr - xl) * 0.5 * ((bl - al) + (br - ar))
    if absolute:
        area_trapezoid = np.abs(area_trapezoid)

    # Hourglass for crossing lines.
    # NaNs should only appear in the degenerate and parallel cases.
    # Those NaNs won't get through the final multiplication so it's ok.
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = intersection((xl, bl), (xr, br), (xl, al), (xr, ar))[0]
    tl_area = 0.5 * (bl - al) * (x_intersect - xl)
    tr_area = 0.5 * (br - ar) * (xr - x_intersect)
    if absolute:
        area_hourglass = np.abs(tl_area) + np.abs(tr_area)
    else:
        area_hourglass = tl_area + tr_area

    # The nan_to_num function allows us to do 0 * nan = 0
    return (1 - cross) * area_trapezoid + cross * np.nan_to_num(area_hourglass)


def intersection(
    p1: Tuple[Numeric, Numeric],
    p2: Tuple[Numeric, Numeric],
    p3: Tuple[Numeric, Numeric],
    p4: Tuple[Numeric, Numeric],
) -> Tuple[Numeric, Numeric]:
    """
    Calculate the intersection of two lines between four points, as defined in
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection.

    This is an array option and works can be used to calculate the intersections of
    entire arrays of points at the same time.

    Args:
        p1: The point (x1, y1), first point of Line 1
        p2: The point (x2, y2), second point of Line 1
        p3: The point (x3, y3), first point of Line 2
        p4: The point (x4, y4), second point of Line 2

    Returns: The point of intersection of the two lines, or (np.nan, np.nan) if the lines are parallel

    """

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D

    return x, y


def mean_fn(tensor, axis=None):
    if isinstance(tensor, torch.Tensor):
        if isinstance(tensor, torch.BoolTensor):
            tensor = tensor.float()
        if axis is None:
            return tensor.mean()
        else:
            return tensor.mean(dim=axis)
    else:
        return np.mean(tensor, axis=axis)


def sum_fn(tensor, axis=None):
    if isinstance(tensor, torch.Tensor):
        if isinstance(tensor, torch.BoolTensor):
            tensor = tensor.float()
        if axis is None:
            return tensor.sum()
        else:
            return tensor.sum(dim=axis)
    else:
        return np.sum(tensor, axis=axis)


def sqrt_fn(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.sqrt()
    else:
        return np.sqrt(tensor)


def abs_fn(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.abs()
    else:
        return np.abs(tensor)


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if array is None:
            array_list.append(array)
            continue
        if isinstance(array, torch.autograd.Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        if not ("keep_list" in kwargs and kwargs["keep_list"]):
            array_list = array_list[0]
    return array_list