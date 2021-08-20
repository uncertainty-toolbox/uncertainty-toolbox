"""
Tests for accuracy metrics.
"""

import pytest
import numpy as np

from uncertainty_toolbox.metrics_accuracy import prediction_error_metrics


@pytest.fixture
def get_test_set():
    y_pred = np.array([1, 2, 3])
    y_std = np.array([0.1, 0.5, 1])
    y_true = np.array([1.25, 2.2, 2.8])
    return y_pred, y_std, y_true


def test_prediction_error_metric_fields(get_test_set):
    """Test if prediction error metrics have correct fields."""
    y_pred, y_std, y_true = get_test_set
    met_dict = prediction_error_metrics(y_pred, y_true)
    met_keys = met_dict.keys()
    assert len(met_keys) == 6

    met_str_list = ["mae", "rmse", "mdae", "marpd", "r2", "corr"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)


def test_prediction_error_metric_values(get_test_set):
    """Test if prediction error metrics have correct values."""
    y_pred, y_std, y_true = get_test_set
    met_dict = prediction_error_metrics(y_pred, y_true)
    print(met_dict)
    assert met_dict["mae"] > 0.21 and met_dict["mae"] < 0.22
    assert met_dict["rmse"] > 0.21 and met_dict["rmse"] < 0.22
    assert met_dict["mdae"] >= 0.20 and met_dict["mdae"] < 0.21
    assert met_dict["marpd"] > 12 and met_dict["marpd"] < 13
    assert met_dict["r2"] > 0.88 and met_dict["r2"] < 0.89
    assert met_dict["corr"] > 0.99 and met_dict["corr"] < 1.0
