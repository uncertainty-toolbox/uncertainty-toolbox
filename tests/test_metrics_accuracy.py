"""
Tests for accuracy metrics.
"""

from uncertainty_toolbox.metrics_accuracy import prediction_error_metrics
from uncertainty_toolbox.data import synthetic_sine_heteroscedastic


predictions, predictions_std, y, x = synthetic_sine_heteroscedastic(10)


def test_prediction_error_metric_fields():
    """Test if prediction error metrics have correct fields."""
    met_dict = prediction_error_metrics(predictions, y)
    met_keys = met_dict.keys()
    assert len(met_keys) == 6

    met_str_list = ["mae", "rmse", "mdae", "marpd", "r2", "corr"]
    bool_list = [s in met_keys for s in met_str_list]
    assert all(bool_list)


def test_prediction_error_metric_values():
    """Test if prediction error metrics have correct values."""
    predictions, predictions_std, y, x = synthetic_sine_heteroscedastic(1000)
    met_dict = prediction_error_metrics(predictions, y)
    assert met_dict['mae'] > 0.3 and met_dict['mae'] < 0.4
    assert met_dict['rmse'] > 0.4 and met_dict['rmse'] < 0.6
    assert met_dict['mdae'] > 0.15 and met_dict['mdae'] < 0.25
    assert met_dict['marpd'] > 60 and met_dict['marpd'] < 70
    assert met_dict['r2'] > 0.55 and met_dict['r2'] < 0.75
    assert met_dict['corr'] > 0.7 and met_dict['corr'] < 0.9
