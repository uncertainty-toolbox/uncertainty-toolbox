"""
Tests for data.
"""

from uncertainty_toolbox.data import (
    synthetic_arange_random,
    synthetic_sine_heteroscedastic,
    curvy_cosine,
)


def test_synthetic_arange_random():
    """Test if correct data is generated."""
    n_list = [10, 20]
    for n in n_list:
        y_pred, y_std, y_true = synthetic_arange_random(n)
        assert len(y_pred) == n
        assert len(y_std) == n
        assert len(y_true) == n
