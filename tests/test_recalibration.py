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
    y_pred, y_std, y_true = supply_test_set
    ma_cal_ratio = optimize_recalibration_ratio(y_pred, y_std, y_true, criterion="ma_cal")
    recal_ma_cal = mean_absolute_calibration_error(
        y_pred, ma_cal_ratio * y_std, y_true
    )
    recal_rms_cal = root_mean_squared_calibration_error(
        y_pred, ma_cal_ratio * y_std, y_true
    )
    recal_miscal = miscalibration_area(
        y_pred, ma_cal_ratio * y_std, y_true
    )
    
    assert np.abs(ma_cal_ratio - 0.33215708813773176) < 1e-6
    assert np.abs(recal_ma_cal - 0.06821616161616162) < 1e-6
    assert np.abs(recal_rms_cal - 0.08800130087804929) < 1e-6
    assert np.abs(recal_miscal - 0.06886262626262629) < 1e-6


def test_rmce_std_recalibration_on_test_set(supply_test_set):
    """
    Test standard deviation recalibration on root mean squared calibration error
    on the test set for some dummy values.
    """
    y_pred, y_std, y_true = supply_test_set
    rms_cal_ratio = optimize_recalibration_ratio(
        y_pred, y_std, y_true, criterion="rms_cal"
    )
    recal_ma_cal = mean_absolute_calibration_error(
        y_pred, rms_cal_ratio * y_std, y_true
    )
    recal_rms_cal = root_mean_squared_calibration_error(
        y_pred, rms_cal_ratio * y_std, y_true
    )
    recal_miscal = miscalibration_area(
        y_pred, rms_cal_ratio * y_std, y_true
    )
    
    assert np.abs(rms_cal_ratio - 0.34900989073212507) < 1e-6
    assert np.abs(recal_ma_cal - 0.06945555555555555) < 1e-6
    assert np.abs(recal_rms_cal - 0.08570902541177935) < 1e-6
    assert np.abs(recal_miscal - 0.07011706864564003) < 1e-6


def test_miscal_area_std_recalibration_on_test_set(supply_test_set):
    """
    Test standard deviation recalibration on miscalibration area
    on the test set for some dummy values.
    """
    y_pred, y_std, y_true = supply_test_set
    miscal_ratio = optimize_recalibration_ratio(
        y_pred, y_std, y_true, criterion="miscal"
    )
    recal_ma_cal = mean_absolute_calibration_error(
        y_pred, miscal_ratio * y_std, y_true
    )
    recal_rms_cal = root_mean_squared_calibration_error(
        y_pred, miscal_ratio * y_std, y_true
    )
    recal_miscal = miscalibration_area(
        y_pred, miscal_ratio * y_std, y_true
    )
    
    assert np.abs(miscal_ratio - 0.3321912522557988) < 1e-6
    assert np.abs(recal_ma_cal - 0.06821616161616162) < 1e-6
    assert np.abs(recal_rms_cal - 0.08800130087804929) < 1e-6
    assert np.abs(recal_miscal - 0.06886262626262629) < 1e-6
