"""
Tests for metrics.
"""

from uncertainty_toolbox.metrics import (
    get_all_accuracy_metrics,
    get_all_average_calibration,
    get_all_adversarial_group_calibration,
    get_all_sharpness_metrics,
    get_all_scoring_rule_metrics,
    get_all_metrics,
)

from uncertainty_toolbox.data import synthetic_sine_heteroscedastic


predictions, predictions_std, y, x = synthetic_sine_heteroscedastic(10)


def test_get_all_accuracy_metrics():
    """Test if all accuracy metrics are returned."""
    met_dict = get_all_accuracy_metrics(predictions, y)
    met_keys = met_dict.keys()
    assert len(met_keys) == 6

    met_str_list = ["mae", "rmse", "mdae", "marpd", "r2", "corr"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)


def test_get_all_sharpness_metrics():
    """Test if all sharpness metrics are returned."""
    met_dict = get_all_sharpness_metrics(predictions_std)
    met_keys = met_dict.keys()
    assert len(met_keys) == 1
    assert "sharp" in met_keys
