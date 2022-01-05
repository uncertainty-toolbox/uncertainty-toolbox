"""
Tests for quantile metrics.
"""

import pytest
import numpy as np

from uncertainty_toolbox.metrics_quantile_models import (
    quantile_accuracy,
    quantile_sharpness,
    quantile_root_mean_squared_calibration_error,
    quantile_mean_absolute_calibration_error,
    quantile_miscalibration_area,
    quantile_check_score,
    quantile_interval_score
)


@pytest.fixture
def get_test_set():
    initial_quantile_predictions = np.array([[1, 1.2, 1.4, 1.6, 1.8, 2], [2, 2.2, 2.4, 2.6, 2.8, 3]])
    initial_quantile_levels = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    y_true = np.array([1.5, 2.5])
    return initial_quantile_predictions, initial_quantile_levels, y_true


def test_prediction_quantile_error_metric_fields(get_test_set):
    """Test if prediction error metrics have correct fields."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    met_dict = quantile_accuracy(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels)
    met_keys = met_dict.keys()
    assert len(met_keys) == 6

    met_str_list = ["mae", "rmse", "mdae", "marpd", "r2", "corr"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)


def test_prediction_quantile_error_metric_values(get_test_set):
    """Test if prediction error metrics have correct values."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    met_dict = quantile_accuracy(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels)
    print(met_dict)
    assert met_dict["mae"] < 1e-3
    assert met_dict["rmse"] < 1e-3
    assert met_dict["mdae"] < 1e-3
    assert met_dict["marpd"] < 1e-3
    assert met_dict["r2"] > 1 - 1e-3
    assert met_dict["corr"] > 1 - 1e-3

def test_prediction_quantile_sharpness(get_test_set):
    """Test if sharpness metric has correct value."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    sharpness = quantile_sharpness(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels)
    assert np.abs(sharpness - 0.95) < 1e-3

def test_prediction_quantile_root_mean_squared_calibration_error(get_test_set):
    """Test if root mean squared calibration error metric has correct value."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    rmsce = quantile_root_mean_squared_calibration_error(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels, num_bins=2)
    assert np.abs(rmsce - 0.01) < 1e-3

def test_prediction_quantile_mean_absolute_calibration_error(get_test_set):
    """Test if mean absolute calibration error metric has correct value."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    mace = quantile_mean_absolute_calibration_error(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels, num_bins=2)
    assert np.abs(mace - 0.01) < 1e-3

def test_prediction_quantile_miscalibration_area(get_test_set):
    """Test if quantile miscalibration area metric has correct value."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    mca = quantile_miscalibration_area(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels, num_bins=2)
    assert np.abs(mca - 0.005) < 1e-3

def test_prediction_quantile_check_score(get_test_set):
    """Test if check score metric has correct value."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    cs = quantile_check_score(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels, num_bins=2)
    assert np.abs(cs - 0.005) < 1e-3

def test_prediction_quantile_interval_score(get_test_set):
    """Test if quantile interval score metric has correct value."""
    initial_quantile_predictions, initial_quantile_levels, y_true = get_test_set
    i_s = quantile_interval_score(y=y_true, method='predictions', initial_quantile_predictions=initial_quantile_predictions, initial_quantile_levels=initial_quantile_levels, num_bins=2)
    assert np.abs(i_s - 0.5) < 1e-3
