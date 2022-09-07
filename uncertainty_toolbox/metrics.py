"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""
from typing import Any, Dict

import numpy as np

from uncertainty_toolbox.metrics_accuracy import prediction_error_metrics
from uncertainty_toolbox.metrics_calibration import (
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    miscalibration_area,
    adversarial_group_calibration,
    sharpness,
)
from uncertainty_toolbox.metrics_scoring_rule import (
    nll_gaussian,
    crps_gaussian,
    check_score,
    interval_score,
)


METRIC_NAMES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mdae": "MDAE",
    "marpd": "MARPD",
    "r2": "R2",
    "corr": "Correlation",
    "rms_cal": "Root-mean-squared Calibration Error",
    "ma_cal": "Mean-absolute Calibration Error",
    "miscal_area": "Miscalibration Area",
    "sharp": "Sharpness",
    "nll": "Negative-log-likelihood",
    "crps": "CRPS",
    "check": "Check Score",
    "interval": "Interval Score",
    "rms_adv_group_cal": ("Root-mean-squared Adversarial Group " "Calibration Error"),
    "ma_adv_group_cal": "Mean-absolute Adversarial Group Calibration Error",
}


def get_all_accuracy_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute all accuracy metrics.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        verbose: Activate verbose mode.

    Returns:
        The evaluations for all accuracy related metrics.
    """
    if verbose:
        print(" (1/n) Calculating accuracy metrics")

    acc_metrics = prediction_error_metrics(y_pred, y_true)
    return acc_metrics


def get_all_average_calibration(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute all metrics for average calibration.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of he predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: The number of bins to use for discretization in some metrics.
        verbose: Activate verbose mode.

    Returns:
        The evaluations for all metrics relating to average calibration.
    """
    if verbose:
        print(" (2/n) Calculating average calibration metrics")

    cali_metrics = {}
    cali_metrics["rms_cal"] = root_mean_squared_calibration_error(
        y_pred, y_std, y_true, num_bins=num_bins
    )
    cali_metrics["ma_cal"] = mean_absolute_calibration_error(
        y_pred, y_std, y_true, num_bins=num_bins
    )
    cali_metrics["miscal_area"] = miscalibration_area(
        y_pred, y_std, y_true, num_bins=num_bins
    )

    return cali_metrics


def get_all_adversarial_group_calibration(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute all metrics for adversarial group calibration.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of he predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: The number of bins to use for discretization in some metrics.
        verbose: Activate verbose mode.

    Returns:
        The evaluations for all metrics relating to adversarial group calibration.
        Each inner dictionary contains the size of each group and the metrics
        computed for each group.
    """
    adv_group_cali_metrics = {}
    if verbose:
        print(" (3/n) Calculating adversarial group calibration metrics")
        print("  [1/2] for mean absolute calibration error")

    ma_adv_group_cali = adversarial_group_calibration(
        y_pred,
        y_std,
        y_true,
        cali_type="mean_abs",
        num_bins=num_bins,
        verbose=verbose,
    )
    ma_adv_group_size = ma_adv_group_cali.group_size
    ma_adv_group_cali_score_mean = ma_adv_group_cali.score_mean
    ma_adv_group_cali_score_stderr = ma_adv_group_cali.score_stderr
    adv_group_cali_metrics["ma_adv_group_cal"] = {
        "group_sizes": ma_adv_group_size,
        "adv_group_cali_mean": ma_adv_group_cali_score_mean,
        "adv_group_cali_stderr": ma_adv_group_cali_score_stderr,
    }

    if verbose:
        print("  [2/2] for root mean squared calibration error")

    rms_adv_group_cali = adversarial_group_calibration(
        y_pred,
        y_std,
        y_true,
        cali_type="root_mean_sq",
        num_bins=num_bins,
        verbose=verbose,
    )
    rms_adv_group_size = rms_adv_group_cali.group_size
    rms_adv_group_cali_score_mean = rms_adv_group_cali.score_mean
    rms_adv_group_cali_score_stderr = rms_adv_group_cali.score_stderr
    adv_group_cali_metrics["rms_adv_group_cal"] = {
        "group_sizes": rms_adv_group_size,
        "adv_group_cali_mean": rms_adv_group_cali_score_mean,
        "adv_group_cali_stderr": rms_adv_group_cali_score_stderr,
    }

    return adv_group_cali_metrics


def get_all_sharpness_metrics(
    y_std: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute all sharpness metrics

    Args:
        y_std: 1D array of he predicted standard deviations for the held out dataset.
        verbose: Activate verbose mode.

    Returns:
        The evaluations for all sharpness metrics.
    """
    if verbose:
        print(" (4/n) Calculating sharpness metrics")

    sharp_metrics = {}
    sharp_metrics["sharp"] = sharpness(y_std)

    return sharp_metrics


def get_all_scoring_rule_metrics(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    resolution: int,
    scaled: bool,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute all scoring rule metrics

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of he predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        resolution: The number of quantiles to use for computation.
        scaled: Whether to scale the score by size of held out set.
        verbose: Activate verbose mode.

    Returns:
        The computed scoring rule metrics.
    """
    if verbose:
        print(" (n/n) Calculating proper scoring rule metrics")

    sr_metrics = {}
    sr_metrics["nll"] = nll_gaussian(y_pred, y_std, y_true, scaled=scaled)
    sr_metrics["crps"] = crps_gaussian(y_pred, y_std, y_true, scaled=scaled)
    sr_metrics["check"] = check_score(
        y_pred, y_std, y_true, scaled=scaled, resolution=resolution
    )
    sr_metrics["interval"] = interval_score(
        y_pred, y_std, y_true, scaled=scaled, resolution=resolution
    )

    return sr_metrics


def _print_adversarial_group_calibration(adv_group_metric_dic, print_group_num=3):

    for a_group_cali_type, a_group_cali_dic in adv_group_metric_dic.items():
        num_groups = a_group_cali_dic["group_sizes"].shape[0]
        print_idxs = [int(x) for x in np.linspace(1, num_groups - 1, print_group_num)]
        print("  {}".format(METRIC_NAMES[a_group_cali_type]))
        for idx in print_idxs:
            print(
                "     Group Size: {:.2f} -- Calibration Error: {:.3f}".format(
                    a_group_cali_dic["group_sizes"][idx],
                    a_group_cali_dic["adv_group_cali_mean"][idx],
                )
            )


def get_all_metrics(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    resolution: int = 99,
    scaled: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute all metrics.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of he predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: The number of bins to use for discretization in some metrics.
        resolution: The number of quantiles to use for computation.
        scaled: Whether to scale the score by size of held out set.
        verbose: Activate verbose mode.

    Returns:
        Dictionary containing all metrics.
    """
    # Accuracy
    accuracy_metrics = get_all_accuracy_metrics(y_pred, y_true, verbose)

    # Calibration
    calibration_metrics = get_all_average_calibration(
        y_pred, y_std, y_true, num_bins, verbose
    )

    # Adversarial Group Calibration
    adv_group_cali_metrics = get_all_adversarial_group_calibration(
        y_pred, y_std, y_true, num_bins, verbose
    )

    # Sharpness
    sharpness_metrics = get_all_sharpness_metrics(y_std, verbose)

    # Proper Scoring Rules
    scoring_rule_metrics = get_all_scoring_rule_metrics(
        y_pred, y_std, y_true, resolution, scaled, verbose
    )

    # Print all outputs
    if verbose:
        print("**Finished Calculating All Metrics**")
        print("\n")
        print(" Accuracy Metrics ".center(60, "="))
        for acc_metric, acc_val in accuracy_metrics.items():
            print("  {:<13} {:.3f}".format(METRIC_NAMES[acc_metric], acc_val))
        print(" Average Calibration Metrics ".center(60, "="))
        for cali_metric, cali_val in calibration_metrics.items():
            print("  {:<37} {:.3f}".format(METRIC_NAMES[cali_metric], cali_val))
        print(" Adversarial Group Calibration Metrics ".center(60, "="))
        _print_adversarial_group_calibration(adv_group_cali_metrics, print_group_num=3)
        print(" Sharpness Metrics ".center(60, "="))
        for sharp_metric, sharp_val in sharpness_metrics.items():
            print("  {:}   {:.3f}".format(METRIC_NAMES[sharp_metric], sharp_val))
        print(" Scoring Rule Metrics ".center(60, "="))
        for sr_metric, sr_val in scoring_rule_metrics.items():
            print("  {:<25} {:.3f}".format(METRIC_NAMES[sr_metric], sr_val))

    all_scores = {
        "accuracy": accuracy_metrics,
        "avg_calibration": calibration_metrics,
        "adv_group_calibration": adv_group_cali_metrics,
        "sharpness": sharpness_metrics,
        "scoring_rule": scoring_rule_metrics,
    }

    return all_scores
