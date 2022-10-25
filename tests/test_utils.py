"""
Test for util functions.
"""

import pytest
import numpy as np

from uncertainty_toolbox.utils import (
    assert_is_flat_same_shape,
    assert_is_positive,
    trapezoid_area,
)


def test_is_flat_same_shape_wrong_type():
    wrong = [1, 2, 3]
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape(wrong)


def test_is_flat_same_shape_wrong_shape():
    wrong = np.arange(9).reshape(3, 3)
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape(wrong)


def test_is_flat_same_shape_wrong_type_and_shape():
    first = np.arange(3)
    wrong = np.arange(3).reshape(1, 3)
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape(first, wrong)


def test_is_flat_same_shape_not_all_same():
    first = np.arange(3)
    wrong = np.arange(5)
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape(first, wrong)


def test_is_flat_same_shape_correct_many_inputs():
    inputs = [np.arange(5) for _ in range(5)]
    assert_is_flat_same_shape(*inputs)


def test_trapezoid_area_convex():
    # convex trapezoid
    _X = np.array([0, 1, 2, 3])
    _A = np.array([0, 0, 0, 0])
    _B = np.array([0, 1, 2, 0])
    xl, al, bl = _X[:-1], _A[:-1], _B[:-1]
    xr, ar, br = _X[1:], _A[1:], _B[1:]
    area_arr = trapezoid_area(xl, al, bl, xr, ar, br)
    assert area_arr.shape == (len(_X) - 1,)
    assert np.sum(area_arr) == pytest.approx(3.0, abs=1e-10)


def test_trapezoid_area_crossed():
    # crossed trapezoid
    _X = np.array([0, 1, 2, 3])
    _A = np.array([0, 0, 0, 0])
    _B = np.array([0, 1, -1, 0])
    xl, al, bl = _X[:-1], _A[:-1], _B[:-1]
    xr, ar, br = _X[1:], _A[1:], _B[1:]
    area_arr = trapezoid_area(xl, al, bl, xr, ar, br)
    assert area_arr.shape == (len(_X) - 1,)
    assert np.sum(area_arr) == pytest.approx(1.5, abs=1e-10)


def test_trapezoid_area_sine_crossed():
    # sine wave, 1 period, crossed trapezoid
    _NUM_DISCRETIZATION = 1000
    _X = np.linspace(0, 2 * np.pi, _NUM_DISCRETIZATION)
    sine_x = np.sin(_X)
    _A = np.zeros_like(_X)
    _B = sine_x
    xl, al, bl = _X[:-1], _A[:-1], _B[:-1]
    xr, ar, br = _X[1:], _A[1:], _B[1:]
    area_arr = trapezoid_area(xl, al, bl, xr, ar, br)
    assert area_arr.shape == (len(_X) - 1,)
    assert np.sum(area_arr) == pytest.approx(4, abs=1e-4)


def test_trapezoid_area_sine_crossed_rotated():
    # sine wave, 5 periods, crossed trapezoid, 45 deg rotated base
    _NUM_DISCRETIZATION = 10000
    _ROT_DEG = -np.pi / 4
    _X = np.linspace(0, 10 * np.pi, _NUM_DISCRETIZATION)
    sine_x = np.sin(_X)
    _A = np.zeros_like(_X)
    _B = sine_x
    line1 = np.stack([_X, _A]).T
    line2 = np.stack([_X, _B]).T
    rot_mat = np.array(
        [[np.cos(_ROT_DEG), -np.sin(_ROT_DEG)], [np.sin(_ROT_DEG), np.cos(_ROT_DEG)]]
    )
    rot_line1 = line1 @ rot_mat
    rot_line2 = line2 @ rot_mat
    x = rot_line1[:, 0]
    a = rot_line1[:, 1]
    b = rot_line2[:, 1]
    xl, al, bl = x[:-1], a[:-1], b[:-1]
    xr, ar, br = x[1:], a[1:], b[1:]
    area_arr = trapezoid_area(xl, al, bl, xr, ar, br)
    assert area_arr.shape == (len(_X) - 1,)
    assert np.sum(area_arr) == pytest.approx(10, abs=1e-4)


def test_is_flat_same_shape_correct_single_input():
    input = np.arange(5)
    assert_is_flat_same_shape(input)


def test_is_flat_same_shape_correct_many_empty_inputs():
    inputs = [np.arange(0) for _ in range(5)]
    assert_is_flat_same_shape(*inputs)


def test_is_flat_same_shape_correct_single_empty_input():
    input = np.arange(0)
    assert_is_flat_same_shape(input)


def test_is_flat_same_shape_empty_call():
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape()


def test_assert_is_positive_wrong_type():
    wrong = [1, 2, 3]
    with pytest.raises(AssertionError):
        assert_is_positive(wrong)


def test_assert_is_positive_with_zero_as_input():
    wrong = np.arange(9)
    with pytest.raises(AssertionError):
        assert_is_positive(wrong)


def test_assert_is_positive_with_negative_inputs():
    wrong = np.arange(-9, 9, 2)
    with pytest.raises(AssertionError):
        assert_is_positive(wrong)


def test_assert_is_positive_correct_many_inputs():
    inputs = [np.arange(1, 9) for _ in range(5)]
    assert_is_positive(*inputs)


def test_assert_is_positive_correct_single_input():
    input = np.arange(1, 9)
    assert_is_positive(input)


def test_assert_is_positive_correct_2D_input():
    input = np.arange(1, 10).reshape(3, 3)
    assert_is_positive(input)


def test_assert_is_positive_empty_call():
    with pytest.raises(AssertionError):
        assert_is_positive()
