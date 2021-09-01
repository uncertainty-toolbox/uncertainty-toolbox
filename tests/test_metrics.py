"""
Tests for metrics.
"""

import pytest
import numpy as np

from uncertainty_toolbox.metrics import (
    get_all_accuracy_metrics,
    get_all_average_calibration,
    get_all_adversarial_group_calibration,
    get_all_sharpness_metrics,
    get_all_scoring_rule_metrics,
    get_all_metrics,
)


@pytest.fixture
def get_test_set():
    y_pred = np.array([1, 2, 3])
    y_std = np.array([0.1, 0.5, 1])
    y_true = np.array([1.5, 3, 2])
    return y_pred, y_std, y_true


def test_get_all_accuracy_metrics_returns(get_test_set):
    """Test if correct accuracy metrics are returned."""
    y_pred, y_std, y_true = get_test_set
    met_dict = get_all_accuracy_metrics(y_pred, y_true)
    met_keys = met_dict.keys()
    assert len(met_keys) == 6

    met_str_list = ["mae", "rmse", "mdae", "marpd", "r2", "corr"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)


def test_get_all_average_calibration_returns(get_test_set):
    """Test if correct average calibration metrics are returned."""
    n_bins = 20
    met_dict = get_all_average_calibration(*get_test_set, n_bins)
    met_keys = met_dict.keys()
    assert len(met_keys) == 3

    met_str_list = ["rms_cal", "ma_cal", "miscal_area"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)


def test_get_all_adversarial_group_calibration_returns(get_test_set):
    """Test if correct adversarial group calibration metrics are returned."""
    n_bins = 20
    met_dict = get_all_adversarial_group_calibration(*get_test_set, n_bins)
    met_keys = met_dict.keys()
    assert len(met_keys) == 2

    met_str_list = ["ma_adv_group_cal", "rms_adv_group_cal"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)

    for met_str in met_str_list:
        inner_dict = met_dict[met_str]
        inner_keys = inner_dict.keys()
        assert len(inner_keys) == 3
        inner_str_list = [
            "group_sizes",
            "adv_group_cali_mean",
            "adv_group_cali_stderr",
        ]
        bool_list = [s in inner_keys for s in inner_str_list]
        assert all(bool_list)


def test_get_all_sharpness_metrics_returns(get_test_set):
    """Test if correct sharpness metrics are returned."""
    y_pred, y_std, y_true = get_test_set
    met_dict = get_all_sharpness_metrics(y_std)
    met_keys = met_dict.keys()
    assert len(met_keys) == 1
    assert "sharp" in met_keys


def test_get_all_scoring_rule_metrics_returns(get_test_set):
    """Test if correct scoring rule metrics are returned."""
    resolution = 99
    scaled = True
    met_dict = get_all_scoring_rule_metrics(*get_test_set, resolution, scaled)
    met_keys = met_dict.keys()
    assert len(met_keys) == 4

    met_str_list = ["nll", "crps", "check", "interval"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)


def test_get_all_metrics_returns(get_test_set):
    """Test if correct metrics are returned by get_all_metrics function."""
    met_dict = get_all_metrics(*get_test_set)
    met_keys = met_dict.keys()
    assert len(met_keys) == 5

    met_str_list = [
        "accuracy",
        "avg_calibration",
        "adv_group_calibration",
        "sharpness",
        "scoring_rule",
    ]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)
