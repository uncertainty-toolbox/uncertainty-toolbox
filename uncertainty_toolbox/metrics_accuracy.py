"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
import torch
from uncertainty_toolbox.utils import assert_is_flat_same_shape


def prediction_error_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Get all prediction error metrics.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.

    Returns:
        A dictionary with Mean average error ('mae'), Root mean squared
        error ('rmse'), Median absolute error ('mdae'),  Mean absolute
        relative percent difference ('marpd'), r^2 ('r2'), and Pearson's
        correlation coefficient ('corr').
    """
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_true)

    # Compute metrics
    if isinstance(y_pred, torch.Tensor):
        mae = (y_true - y_pred).abs().mean().item()
        rmse = (y_pred - y_true).square().mean().sqrt().item()
        mdae = (y_true - y_pred).abs().median().item()
        residuals = y_true - y_pred
        marpd = torch.abs(2 * residuals / (torch.abs(y_pred) + torch.abs(y_true))).mean().item() * 100
        r2 = r2_score(y_true.cpu(), y_pred.cpu())
        corr = torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1].item()
    else:
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
