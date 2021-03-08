"""
Tests for scoring rule metrics.
"""
import numpy as np
import pytest

from uncertainty_toolbox.metrics_scoring_rule import (
    nll_gaussian,
    crps_gaussian,
    check_score,
    interval_score,
)


@pytest.fixture
def supply_test_set():
    y_pred = np.array([1, 2, 3])
    y_std = np.array([0.1, 0.5, 1])
    y_true = np.array([1.5, 3, 2])
    return y_pred, y_std, y_true


def test_nll_gaussian_on_test_set(supply_test_set):
    """Test Gaussian NLL on the test set for some dummy values."""
    assert np.abs(nll_gaussian(*supply_test_set) - 4.920361108686675) < 1e-6


def test_nll_gaussian_on_one_pt():
    """Sanity check by testing one point at mean of gaussian."""
    y_pred = np.array([0])
    y_true = np.array([0])
    y_std = np.array([1 / np.sqrt(2 * np.pi)])
    assert np.abs(nll_gaussian(y_pred, y_std, y_true)) < 1e-6


def test_crps_gaussian_on_test_set(supply_test_set):
    """Test CRPS on the test set for some dummy values."""
    assert np.abs(crps_gaussian(*supply_test_set) - 0.59080610693) < 1e-6


def test_check_score_on_test_set(supply_test_set):
    """Test check score on the test set for some dummy values."""
    assert np.abs(check_score(*supply_test_set) - 0.29801437323836477) < 1e-6


def test_check_score_on_one_pt():
    """Sanity check to show that check score is minimized (i.e. 0) if data
    occurs at the exact requested quantile."""
    y_pred = np.array([0])
    y_true = np.array([1])
    y_std = np.array([1])
    score = check_score(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        start_q=0.5 + 0.341,
        end_q=0.5 + 0.341,
        resolution=1,
    )
    assert np.abs(score) < 1e-2


def test_interval_score_on_test_set(supply_test_set):
    """Test interval score on the test set for some dummy values."""
    assert np.abs(interval_score(*supply_test_set) - 3.20755700861995) < 1e-6


def test_interval_score_on_one_pt():
    """Sanity check on interval score. For one point in the center of the
    distribution and intervals one standard deviation and two standard
    deviations away, should return ((1 std) * 2 + (2 std) * 2) / 2 = 3.
    """
    y_pred = np.array([0])
    y_true = np.array([0])
    y_std = np.array([1])
    score = interval_score(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        start_p=0.682,
        end_p=0.954,
        resolution=2,
    )
    assert np.abs(score - 3) < 1e-2
