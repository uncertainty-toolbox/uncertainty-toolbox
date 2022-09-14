"""
Tests for recalibration procedures.
"""
import random

import numpy as np
import pytest


from uncertainty_toolbox.recalibration import (
    iso_recal,
    optimize_recalibration_ratio,
    get_std_recalibrator,
    get_quantile_recalibrator,
    get_interval_recalibrator,
)
from uncertainty_toolbox.metrics_calibration import (
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    miscalibration_area,
    get_proportion_lists_vectorized,
    get_prediction_interval,
    get_quantile,
)


@pytest.fixture
def supply_test_set():
    y_pred = np.arange(100) / 100.0
    y_std = np.arange(1, 101) / 20.0
    y_true = np.arange(100) / 100.0 + 0.5
    return y_pred, y_std, y_true


def test_recal_model_mace_criterion_on_test_set(supply_test_set):
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
    assert np.abs(test_mace - 0.24206060606060598) < 1e-3
    assert np.abs(recal_test_mace - 0.003035353535353514) < 1e-3
    for idx in range(1, recal_exp_props.shape[0]):
        assert recal_exp_props[idx - 1] <= recal_exp_props[idx]


def test_recal_model_rmce_criterion_on_test_set(supply_test_set):
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

    assert np.abs(test_rmsce - 0.28741418862839013) < 1e-3
    assert np.abs(recal_test_rmsce - 0.003981861230030349) < 1e-3
    for idx in range(1, recal_exp_props.shape[0]):
        assert recal_exp_props[idx - 1] <= recal_exp_props[idx]


def test_recal_model_miscal_area_criterion_on_test_set(supply_test_set):
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

    assert np.abs(test_miscal_area - 0.24426139657444004) < 1e-3
    assert np.abs(recal_test_miscal_area - 0.0029569160997732244) < 1e-3
    for idx in range(1, recal_exp_props.shape[0]):
        assert recal_exp_props[idx - 1] <= recal_exp_props[idx]


def test_optimize_recalibration_ratio_mace_criterion(supply_test_set):
    """
    Test standard deviation recalibration on mean absolute calibration error
    on the test set for some dummy values.
    """
    random.seed(0)
    np.random.seed(seed=0)

    y_pred, y_std, y_true = supply_test_set
    ma_cal_ratio = optimize_recalibration_ratio(
        y_pred, y_std, y_true, criterion="ma_cal"
    )
    recal_ma_cal = mean_absolute_calibration_error(y_pred, ma_cal_ratio * y_std, y_true)
    recal_rms_cal = root_mean_squared_calibration_error(
        y_pred, ma_cal_ratio * y_std, y_true
    )
    recal_miscal = miscalibration_area(y_pred, ma_cal_ratio * y_std, y_true)

    assert np.abs(ma_cal_ratio - 0.33215708813773176) < 1e-3
    assert np.abs(recal_ma_cal - 0.06821616161616162) < 1e-3
    assert np.abs(recal_rms_cal - 0.08800130087804929) < 1e-3
    assert np.abs(recal_miscal - 0.06886262626262629) < 1e-3


def test_optimize_recalibration_ratio_rmce_criterion(supply_test_set):
    """
    Test standard deviation recalibration on root mean squared calibration error
    on the test set for some dummy values.
    """
    random.seed(0)
    np.random.seed(seed=0)

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
    recal_miscal = miscalibration_area(y_pred, rms_cal_ratio * y_std, y_true)

    assert np.abs(rms_cal_ratio - 0.34900989073212507) < 1e-3
    assert np.abs(recal_ma_cal - 0.06945555555555555) < 1e-3
    assert np.abs(recal_rms_cal - 0.08570902541177935) < 1e-3
    assert np.abs(recal_miscal - 0.07011706864564003) < 1e-3


def test_optimize_recalibration_ratio_miscal_area_criterion(supply_test_set):
    """
    Test standard deviation recalibration on miscalibration area
    on the test set for some dummy values.
    """
    random.seed(0)
    np.random.seed(seed=0)

    y_pred, y_std, y_true = supply_test_set
    miscal_ratio = optimize_recalibration_ratio(
        y_pred, y_std, y_true, criterion="miscal"
    )
    recal_ma_cal = mean_absolute_calibration_error(y_pred, miscal_ratio * y_std, y_true)
    recal_rms_cal = root_mean_squared_calibration_error(
        y_pred, miscal_ratio * y_std, y_true
    )
    recal_miscal = miscalibration_area(y_pred, miscal_ratio * y_std, y_true)

    assert np.abs(miscal_ratio - 0.3321912522557988) < 1e-3
    assert np.abs(recal_ma_cal - 0.06821616161616162) < 1e-3
    assert np.abs(recal_rms_cal - 0.08800130087804929) < 1e-3
    assert np.abs(recal_miscal - 0.06886262626262629) < 1e-3


def test_get_prediction_interval_recalibrated(supply_test_set):
    """
    Test standard deviation recalibration on miscalibration area
    on the test set for some dummy values.
    """
    random.seed(0)
    np.random.seed(seed=0)

    y_pred, y_std, y_true = supply_test_set
    test_exp_props, test_obs_props = get_proportion_lists_vectorized(
        y_pred, y_std, y_true, num_bins=100, recal_model=None
    )
    recal_model = iso_recal(test_exp_props, test_obs_props)

    test_quantile_prop_list = [
        (0.01, 0.0, 0.0),
        (0.25, 0.69, 0.25),
        (0.5, 0.86, 0.5),
        (0.75, 0.92, 0.75),
        (0.99, 0.97, 0.97),
    ]

    for (q, test_orig_prop, test_recal_prop) in test_quantile_prop_list:
        orig_bounds = get_prediction_interval(y_pred, y_std, q, None)
        recal_bounds = get_prediction_interval(y_pred, y_std, q, recal_model)

        orig_prop = np.mean(
            (orig_bounds.lower <= y_true) * (y_true <= orig_bounds.upper)
        )
        recal_prop = np.mean(
            (recal_bounds.lower <= y_true) * (y_true <= recal_bounds.upper)
        )

        assert np.max(np.abs(test_orig_prop - orig_prop)) < 1e-3
        assert np.max(np.abs(test_recal_prop - recal_prop)) < 1e-3


def test_get_std_recalibrator(supply_test_set):
    """
    Test get_std_recalibration on the test set for some dummy values.
    """
    random.seed(0)
    np.random.seed(seed=0)

    y_pred, y_std, y_true = supply_test_set

    test_quantile_prop_list = [
        (0.01, 0.00, 0.00),
        (0.25, 0.06, 0.00),
        (0.50, 0.56, 0.00),
        (0.75, 0.74, 0.56),
        (0.99, 0.89, 0.88),
    ]

    std_recalibrator = get_std_recalibrator(y_pred, y_std, y_true)

    for (q, test_prop_in_pi, test_prop_under_q) in test_quantile_prop_list:
        y_std_recal = std_recalibrator(y_std)
        pi = get_prediction_interval(y_pred, y_std_recal, q)
        prop_in_pi = ((pi.lower <= y_true) * (y_true <= pi.upper)).mean()
        quantile_bound = get_quantile(y_pred, y_std_recal, q)
        prop_under_q = (quantile_bound >= y_true).mean()
        assert np.max(np.abs(test_prop_in_pi - prop_in_pi)) < 1e-3
        assert np.max(np.abs(test_prop_under_q - prop_under_q)) < 1e-3


def test_get_quantile_recalibrator(supply_test_set):
    """
    Test get_std_recalibration on the test set for some dummy values.
    """
    random.seed(0)
    np.random.seed(seed=0)

    y_pred, y_std, y_true = supply_test_set

    test_quantile_prop_list = [
        (0.01, 0.00),
        (0.25, 0.00),
        (0.50, 0.00),
        (0.75, 0.00),
        (0.99, 0.83),
    ]

    quantile_recalibrator = get_quantile_recalibrator(y_pred, y_std, y_true)

    for (q, test_prop_under_q) in test_quantile_prop_list:
        quantile_bound_recal = quantile_recalibrator(y_pred, y_std, q)
        assert all(np.isfinite(quantile_bound_recal))
        prop_under_q_recal = (quantile_bound_recal >= y_true).mean()
        assert np.max(np.abs(test_prop_under_q - prop_under_q_recal)) < 1e-3


def test_get_interval_recalibrator(supply_test_set):
    """
    Test get_std_recalibration on the test set for some dummy values.
    """
    random.seed(0)
    np.random.seed(seed=0)

    y_pred, y_std, y_true = supply_test_set

    test_quantile_prop_list = [
        (0.01, 0.00),
        (0.25, 0.25),
        (0.50, 0.50),
        (0.75, 0.75),
        (0.99, 0.97),
    ]

    interval_recalibrator = get_interval_recalibrator(y_pred, y_std, y_true)

    for (q, test_prop_in_interval) in test_quantile_prop_list:
        interval_recal = interval_recalibrator(y_pred, y_std, q)
        prop_in_interval_recal = (
            (interval_recal.lower <= y_true) * (y_true <= interval_recal.upper)
        ).mean()
        assert np.max(np.abs(test_prop_in_interval - prop_in_interval_recal)) < 1e-3
