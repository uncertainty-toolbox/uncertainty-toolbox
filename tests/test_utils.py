"""
Test for util functions.
"""

import pytest
import numpy as np

from uncertainty_toolbox.utils import assert_is_flat_same_shape, assert_is_positive


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
