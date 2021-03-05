"""
Tests for calibration metrics.
"""
import numpy as np
import pytest

from uncertainty_toolbox.metrics_calibration import (
    sharpness,
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    adversarial_group_calibration,
    miscalibration_area,
    get_proportion_lists_vectorized,
    get_proportion_lists,
    get_proportion_in_interval,
)


@pytest.fixture
def supply_test_set():
    y_pred = np.array([1, 2, 3])
    y_std = np.array([0.1, 0.5, 1])
    y_true = np.array([1.5, 3, 2])
    return y_pred, y_std, y_true


def test_sharpness_on_test_set(supply_test_set):
    """Test sharpness on the test set for some dummy values."""
    _, test_std, _ = supply_test_set
    assert np.abs(sharpness(test_std) - 0.648074069840786) < 1e-6


def test_get_proportion_lists_on_test_set(supply_test_set):
    """Test get_proportion_lists on the test set for some dummy values."""
    test_exp_props, test_obs_props = get_proportion_lists(
        *supply_test_set, num_bins=100, recal_model=None
    )
    assert (
        np.max(np.abs(np.unique(test_exp_props) - np.linspace(0, 1, 100)))
        < 1e-6
    )
    assert (
        np.max(
            np.abs(
                np.sort(np.unique(test_obs_props)) - np.array([0.0, 0.33333333, 0.66666667, 1.0])
            )
        )
        < 1e-6
    )


def get_proportion_lists_vectorized_on_test_set(supply_test_set):
    """Test get_proportion_lists_vectorized on the test set for some dummy values."""
    test_exp_props, test_obs_props = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None
    )
    assert (
        np.max(np.abs(np.unique(test_exp_props) - np.linspace(0, 1, 100)))
        < 1e-6
    )
    assert (
        np.max(
            np.abs(
                test_obs_props - np.array([0.0, 0.33333333, 0.66666667, 1.0])
            )
        )
        < 1e-6
    )


def get_proportion_in_interval_on_test_set(supply_test_set):
    """Test get_proportion_in_interval on the test set for some dummy values."""
    test_quantile_value_list = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.3333333333333333),
        (1.0, 1.0),
    ]
    for (test_q, test_val) in test_quantile_value_list:
        assert (
            np.abs(
                get_proportion_in_interval(*supply_test_set, quantile=test_q)
                - test_val
            )
            < 1e-6
        )


def test_vectorization_for_proportion_list_on_test_set(supply_test_set):
    """Test vectorization in get_proportion_lists on the test set for some dummy values."""
    test_exp_props_nonvec, test_obs_props_nonvec = get_proportion_lists(
        *supply_test_set, num_bins=100, recal_model=None
    )

    test_exp_props_vec, test_obs_props_vec = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None
    )
    assert np.max(np.abs(test_exp_props_nonvec - test_exp_props_vec)) < 1e-6
    assert np.max(np.abs(test_obs_props_nonvec - test_obs_props_vec)) < 1e-6


def test_rms_calibration_error_on_test_set(supply_test_set):
    """Test root mean squared calibration error on some dummy values."""
    test_rmsce_nonvectorized = root_mean_squared_calibration_error(
        *supply_test_set, num_bins=100, vectorized=False, recal_model=None
    )
    test_rmsce_vectorized = root_mean_squared_calibration_error(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=None
    )
    assert np.abs(test_rmsce_nonvectorized - test_rmsce_vectorized) < 1e-6
    assert np.abs(test_rmsce_vectorized - 0.4165757476562379) < 1e-6


def test_ma_calibration_error_on_test_set(supply_test_set):
    """Test mean absolute calibration error on some dummy values."""
    test_mace_nonvectorized = mean_absolute_calibration_error(
        *supply_test_set, num_bins=100, vectorized=False, recal_model=None
    )
    test_mace_vectorized = mean_absolute_calibration_error(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=None
    )
    assert np.abs(test_mace_nonvectorized - test_mace_vectorized) < 1e-6
    assert np.abs(test_mace_vectorized - 0.3733333333333335) < 1e-6


def test_miscalibration_area_on_test_set(supply_test_set):
    """Test miscalibration area on some dummy values."""
    test_miscal_area_nonvectorized = miscalibration_area(
        *supply_test_set, num_bins=100, vectorized=False, recal_model=None
    )
    test_miscal_area_vectorized = miscalibration_area(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=None
    )
    assert (
        np.abs(test_miscal_area_nonvectorized - test_miscal_area_vectorized)
        < 1e-6
    )
    assert np.abs(test_miscal_area_vectorized - 0.37710437710437716) < 1e-6


def test_adversarial_group_calibration_on_test_set(supply_test_set):
    """Test adversarial group calibration on test set for some dummy values."""
    test_out = adversarial_group_calibration(
        *supply_test_set,
        cali_type="mean_abs",
        num_bins=100,
        num_group_bins=10,
        draw_with_replacement=False,
        num_trials=10,
        num_group_draws=10,
        verbose=False
    )

    print(test_out.score_mean)
    assert np.max(np.abs(test_out.group_size - np.linspace(0, 1, 10))) < 1e-6
    assert test_out.score_mean[0] < 0.5
    assert np.abs(test_out.score_mean[-1] - 0.37333333) < 1e-6
    assert np.min(test_out.score_stderr) >= 0
