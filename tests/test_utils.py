"""
Test for util functions.
"""

import pytest
import numpy as np

from uncertainty_toolbox.utils import assert_is_flat_same_shape

def test_is_flat_same_shape_wrong_type():
    wrong = [1, 2, 3]
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape(wrong)

def test_is_flat_same_shape_wrong_shape():
    wrong = np.arange(9).reshape(3, 3)
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape(wrong)

def test_is_flat_same_shape_not_all_same():
    first = np.arange(3)
    wrong = np.arange(5)
    with pytest.raises(AssertionError):
        assert_is_flat_same_shape(first, wrong)

def test_is_flat_same_shape_correct():
    inputs = [np.arange(5) for _ in range(5)]
    assert_is_flat_same_shape(*inputs)
