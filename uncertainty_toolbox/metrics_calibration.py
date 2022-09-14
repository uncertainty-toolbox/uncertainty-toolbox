"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""
from typing import Any, Tuple, Optional
from argparse import Namespace

import numpy as np
from scipy import stats
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from uncertainty_toolbox.utils import (
    assert_is_flat_same_shape,
    assert_is_positive,
    trapezoid_area,
)


def sharpness(y_std: np.ndarray) -> float:
    """Return sharpness (a single measure of the overall confidence).

    Args:
        y_std: 1D array of the predicted standard deviations for the held out dataset.

    Returns:
        A single scalar which quantifies the average of the standard deviations.
    """
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_std)
    # Check that input std is positive
    assert_is_positive(y_std)

    # Compute sharpness
    sharp_metric = np.sqrt(np.mean(y_std**2))

    return sharp_metric


def root_mean_squared_calibration_error(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    vectorized: bool = False,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> float:
    """Root mean squared calibration error.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        vectorized: whether to vectorize computation for observed proportions.
                    (while setting to True is faster, it has much higher memory requirements
                    and may fail to run for larger datasets).
        recal_model: an sklearn isotonic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.

    Returns:
        A single scalar which calculates the root mean squared calibration error.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce


def mean_absolute_calibration_error(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    vectorized: bool = False,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> float:
    """Mean absolute calibration error; identical to ECE.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        vectorized: whether to vectorize computation for observed proportions.
                    (while setting to True is faster, it has much higher memory requirements
                    and may fail to run for larger datasets).
        recal_model: an sklearn isotonic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.

    Returns:
        A single scalar which calculates the mean absolute calibration error.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace


def adversarial_group_calibration(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    cali_type: str,
    prop_type: str = "interval",
    num_bins: int = 100,
    num_group_bins: int = 10,
    draw_with_replacement: bool = False,
    num_trials: int = 10,
    num_group_draws: int = 10,
    verbose: bool = False,
) -> Namespace:
    """Adversarial group calibration.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        cali_type: type of calibration error to measure; one of ["mean_abs", "root_mean_sq"].
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
        num_bins: number of discretizations for the probability space [0, 1].
        num_group_bins: number of discretizations for group size proportions between 0 and 1.
        draw_with_replacement: True to draw subgroups that draw from the dataset with replacement.
        num_trials: number of trials to estimate the worst calibration error per group size.
        num_group_draws: number of subgroups to draw per given group size to measure calibration error on.
        verbose: True to print progress statements.

    Returns:
        A Namespace with an array of the group sizes, the mean of the worst
        calibration errors for each group size, and the standard error of the
        worst calibration error for each group size
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    num_pts = y_true.shape[0]

    if cali_type == "mean_abs":
        cali_fn = mean_absolute_calibration_error
    elif cali_type == "root_mean_sq":
        cali_fn = root_mean_squared_calibration_error

    num_pts = y_std.shape[0]
    ratio_arr = np.linspace(0, 1, num_group_bins)
    score_mean_per_ratio = []
    score_stderr_per_ratio = []
    if verbose:
        print(
            (
                "Measuring adversarial group calibration by spanning group"
                " size between {} and {}, in {} intervals"
            ).format(np.min(ratio_arr), np.max(ratio_arr), num_group_bins)
        )
    progress = tqdm(ratio_arr) if verbose else ratio_arr
    for r in progress:
        group_size = max([int(round(num_pts * r)), 2])
        score_per_trial = []  # list of worst miscalibrations encountered
        for _ in range(num_trials):
            group_miscal_scores = []
            for g_idx in range(num_group_draws):
                rand_idx = np.random.choice(
                    num_pts, group_size, replace=draw_with_replacement
                )
                group_y_pred = y_pred[rand_idx]
                group_y_true = y_true[rand_idx]
                group_y_std = y_std[rand_idx]
                group_miscal = cali_fn(
                    group_y_pred,
                    group_y_std,
                    group_y_true,
                    num_bins=num_bins,
                    vectorized=True,
                    prop_type=prop_type,
                )
                group_miscal_scores.append(group_miscal)
            max_miscal_score = np.max(group_miscal_scores)
            score_per_trial.append(max_miscal_score)
        score_mean_across_trials = np.mean(score_per_trial)
        score_stderr_across_trials = np.std(score_per_trial, ddof=1)
        score_mean_per_ratio.append(score_mean_across_trials)
        score_stderr_per_ratio.append(score_stderr_across_trials)

    out = Namespace(
        group_size=ratio_arr,
        score_mean=np.array(score_mean_per_ratio),
        score_stderr=np.array(score_stderr_per_ratio),
    )
    return out


def miscalibration_area(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    vectorized: bool = False,
    recal_model: Any = None,
    prop_type: str = "interval",
) -> float:
    """Miscalibration area.

    This is identical to mean absolute calibration error and ECE, however
    the integration here is taken by tracing the area between curves.
    In the limit of num_bins, miscalibration area and
    mean absolute calibration error will converge to the same value.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        vectorized: whether to vectorize computation for observed proportions.
                    (while setting to True is faster, it has much higher memory requirements
                    and may fail to run for larger datasets).
        recal_model: an sklearn isotonic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.

    Returns:
        A single scalar which calculates the miscalibration area.
    """
    # Compute the expected proportions and the residuals.
    exp_proportions = np.linspace(0, 1, num_bins)
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions
    residuals = y_pred - y_true

    # Get the inverse of the CDF at each of these depending on the prop_type.
    if prop_type == "interval":
        expected_sd_multiples = stats.norm(0, 1).ppf(0.5 + in_exp_proportions / 2.0)
        sd_multiples = np.abs(residuals) / y_std
    elif prop_type == "quantile":
        expected_sd_multiples = stats.norm(0, 1).ppf(in_exp_proportions)
        sd_multiples = residuals / y_std
    else:
        raise ValueError(f"Unknown prop_type {prop_type}")

    # For each bin edge, see how many of our data points deviate less than the
    # corresponding sd multiple.
    if vectorized:
        obs_proportions = (sd_multiples.reshape(-1, 1) <= expected_sd_multiples).mean(0)
    else:
        obs_proportions = np.array(
            [
                np.mean(sd_multiples <= expected_sd_multiples[i])
                for i in range(len(expected_sd_multiples))
            ]
        )

    # Now calculate the area between these and the line y=x.
    areas = trapezoid_area(
        exp_proportions[:-1],
        exp_proportions[:-1],
        obs_proportions[:-1],
        exp_proportions[1:],
        exp_proportions[1:],
        obs_proportions[1:],
        absolute=True,
    )
    return areas.sum()


def get_proportion_lists_vectorized(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    recal_model: Any = None,
    prop_type: str = "interval",
) -> Tuple[np.ndarray, np.ndarray]:
    """Arrays of expected and observed proportions

    Returns the expected proportions and observed proportion of points falling into
    intervals corresponding to a range of quantiles.
    Computations here are vectorized for faster execution, but this function is
    not suited when there are memory constraints.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        recal_model: an sklearn isotonic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.

    Returns:
        A tuple of two numpy arrays, expected proportions and observed proportions

    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    norm = stats.norm(loc=0, scale=1)
    if prop_type == "interval":
        gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)
        gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)

        above_lower = normalized_residuals >= gaussian_lower_bound
        below_upper = normalized_residuals <= gaussian_upper_bound

        within_quantile = above_lower * below_upper
        obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)
    elif prop_type == "quantile":
        gaussian_quantile_bound = norm.ppf(in_exp_proportions)
        below_quantile = normalized_residuals <= gaussian_quantile_bound
        obs_proportions = np.sum(below_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions


def get_proportion_lists(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> Tuple[np.ndarray, np.ndarray]:
    """Arrays of expected and observed proportions

    Return arrays of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    Computations here are not vectorized, in case there are memory constraints.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        recal_model: an sklearn isotonic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.

    Returns:
        A tuple of two numpy arrays, expected proportions and observed proportions
    """
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    if prop_type == "interval":
        obs_proportions = [
            get_proportion_in_interval(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]
    elif prop_type == "quantile":
        obs_proportions = [
            get_proportion_under_quantile(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]

    return exp_proportions, obs_proportions


def get_proportion_in_interval(
    y_pred: np.ndarray, y_std: np.ndarray, y_true: np.ndarray, quantile: float
) -> float:
    """For a specified quantile, return the proportion of points falling into
    an interval corresponding to that quantile.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        quantile: a specified quantile level

    Returns:
        A single scalar which is the proportion of the true labels falling into the
        prediction interval for the specified quantile.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    lower_bound = norm.ppf(0.5 - quantile / 2)
    upper_bound = norm.ppf(0.5 + quantile / 2)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true

    normalized_residuals = residuals.reshape(-1) / y_std.reshape(-1)

    num_within_quantile = 0
    for resid in normalized_residuals:
        if lower_bound <= resid and resid <= upper_bound:
            num_within_quantile += 1.0
    proportion = num_within_quantile / len(residuals)

    return proportion


def get_proportion_under_quantile(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    quantile: float,
) -> float:
    """Get the proportion of data that are below the predicted quantile.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        quantile: The quantile level to check.

    Returns:
        The proportion of data below the quantile level.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    quantile_bound = norm.ppf(quantile)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true

    normalized_residuals = residuals / y_std

    num_below_quantile = 0
    for resid in normalized_residuals:
        if resid <= quantile_bound:
            num_below_quantile += 1.0
    proportion = num_below_quantile / len(residuals)

    return proportion


def get_prediction_interval(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    quantile: np.ndarray,
    recal_model: Optional[IsotonicRegression] = None,
) -> Namespace:
    """Return the centered predictional interval corresponding to a quantile.

    For a specified quantile level q (must be a float, or a singleton),
    return the centered prediction interval corresponding
    to the pair of quantiles at levels (0.5-q/2) and (0.5+q/2),
    i.e. interval that has nominal coverage equal to q.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        quantile: The quantile level to check.
        recal_model: A recalibration model to apply before computing the interval.

    Returns:
        Namespace containing the lower and upper bound corresponding to the
        centered interval.
    """

    if isinstance(quantile, float):
        quantile = np.array([quantile])

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std)
    assert_is_flat_same_shape(quantile)
    assert quantile.size == 1
    # Check that input std is positive
    assert_is_positive(y_std)

    if not np.logical_and((0.0 < quantile.item()), (quantile.item() < 1.0)):
        raise ValueError("Quantile must be greater than 0.0 and less than 1.0")

    # if recal_model is not None, calculate recalibrated quantile
    if recal_model is not None:
        quantile = recal_model.predict(quantile)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=y_pred, scale=y_std)
    lower_bound = norm.ppf(0.5 - quantile / 2)
    upper_bound = norm.ppf(0.5 + quantile / 2)

    bounds = Namespace(
        upper=upper_bound,
        lower=lower_bound,
    )

    return bounds


def get_quantile(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    quantile: np.ndarray,
    recal_model: Optional[IsotonicRegression] = None,
) -> float:
    """Return the value corresponding with a quantile.

    For a specified quantile level q (must be a float, or a singleton),
    return the quantile prediction,
    i.e. bound that has nominal coverage below the bound equal to q.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        quantile: The quantile level to check.
        recal_model: A recalibration model to apply before computing the interval.

    Returns:
        The value at which the quantile is achieved.
    """
    if isinstance(quantile, float):
        quantile = np.array([quantile])

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std)
    assert_is_flat_same_shape(quantile)
    assert quantile.size == 1
    # Check that input std is positive
    assert_is_positive(y_std)

    if not np.logical_and((0.0 < quantile.item()), (quantile.item() < 1.0)):
        raise ValueError("Quantile must be greater than 0.0 and less than 1.0")

    # if recal_model is not None, calculate recalibrated quantile
    if recal_model is not None:
        quantile = recal_model.predict(quantile)

    # Computer quantile bound
    norm = stats.norm(loc=y_pred, scale=y_std)
    quantile_prediction = norm.ppf(quantile).flatten()

    return quantile_prediction
