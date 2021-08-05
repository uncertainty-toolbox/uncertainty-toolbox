"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from uncertainty_toolbox.utils import assert_is_flat_same_shape


def prediction_error_metrics(y_pred, y_true):
    """
    Return prediction error metrics as a dict with keys:
    - Mean average error ('mae')
    - Root mean squared error ('rmse')
    - Median absolute error ('mdae')
    - Mean absolute relative percent difference ('marpd')
    - r^2 ('r2')
    - Pearson's correlation coefficient ('corr')
    """
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_true)

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mdae = median_absolute_error(y_true, y_pred)
    residuals = y_true - y_pred
    marpd = np.abs(2 * residuals / (np.abs(y_pred) + np.abs(y_true))).mean() * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    prediction_metrics = {
        "mae": mae,
        "rmse": rmse,
        "mdae": mdae,
        "marpd": marpd,
        "r2": r2,
        "corr": corr,
    }

    return prediction_metrics
