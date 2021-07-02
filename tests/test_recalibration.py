"""
Tests for recalibration procedures.
"""
import numpy as np
import pytest

from uncertainty_toolbox.recalibration import (
    iso_recal, 
    optimize_recalibration_ratio,
)
from uncertainty_toolbox.metrics_calibration import (
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    miscalibration_area,
    get_proportion_lists_vectorized,
)


@pytest.fixture
def supply_test_set():
    y_pred = np.arange(100) / 100.0
    y_std = np.arange(1, 101) / 20.0
    y_true = np.arange(100) / 100.0 + 0.5
    return y_pred, y_std, y_true


def test_mace_recalibration_on_test_set(supply_test_set):
    """
    Test recalibration on mean absolute calibration error on the test set
    for some dummy values.
    """
    test_mace = mean_absolute_calibration_error(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=None
    )
    test_exp_props, test_obs_props = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None
    )
    recal_model = iso_recal(test_exp_props, test_obs_props)
    recal_test_mace = mean_absolute_calibration_error(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=recal_model
    )
    recal_exp_props = recal_model.predict(test_obs_props)
    assert np.abs(test_mace - 0.24206060606060598) < 1e-6
    assert np.abs(recal_test_mace - 0.003035353535353514) < 1e-6
    for idx in range(1, recal_exp_props.shape[0]):
        assert recal_exp_props[idx - 1] <= recal_exp_props[idx]


def test_rmce_recalibration_on_test_set(supply_test_set):
    """
    Test recalibration on root mean squared calibration error on the test set
    for some dummy values.
    """
    test_rmsce = root_mean_squared_calibration_error(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=None
    )
    test_exp_props, test_obs_props = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None
    )
    recal_model = iso_recal(test_exp_props, test_obs_props)
    recal_test_rmsce = root_mean_squared_calibration_error(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=recal_model
    )
    recal_exp_props = recal_model.predict(test_obs_props)

    assert np.abs(test_rmsce - 0.28741418862839013) < 1e-6
    assert np.abs(recal_test_rmsce - 0.003981861230030349) < 1e-6
    for idx in range(1, recal_exp_props.shape[0]):
        assert recal_exp_props[idx - 1] <= recal_exp_props[idx]


def test_miscal_area_recalibration_on_test_set(supply_test_set):
    """
    Test recalibration on miscalibration area on the test set
    for some dummy values.
    """
    test_miscal_area = miscalibration_area(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=None
    )
    test_exp_props, test_obs_props = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None
    )
    recal_model = iso_recal(test_exp_props, test_obs_props)
    recal_test_miscal_area = miscalibration_area(
        *supply_test_set, num_bins=100, vectorized=True, recal_model=recal_model
    )
    recal_exp_props = recal_model.predict(test_obs_props)

    assert np.abs(test_miscal_area - 0.24426139657444004) < 1e-6
    assert np.abs(recal_test_miscal_area - 0.0029569160997732057) < 1e-6
    for idx in range(1, recal_exp_props.shape[0]):
        assert recal_exp_props[idx - 1] <= recal_exp_props[idx]


def test_mace_std_recalibration_on_test_set(supply_test_set):
    """
    Test standard deviation recalibration on mean absolute calibration error
    on the test set for some dummy values.
    """
    pass


def test_rmce_std_recalibration_on_test_set(supply_test_set):
    """
    Test standard deviation recalibration on root mean squared calibration error
    on the test set for some dummy values.
    """
    pass


def test_miscal_area_std_recalibration_on_test_set(supply_test_set):
    """
    Test standard deviation recalibration on miscalibration area
    on the test set for some dummy values.
    """
    pass
