"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""

import numpy as np
import tqdm
from scipy import stats
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union


def sharpness(y_std):
    """
    Return sharpness (a single measure of the overall confidence).
    """

    # Compute sharpness
    sharpness = np.sqrt(np.mean(y_std ** 2))

    return sharpness


def root_mean_squared_calibration_error(
    y_pred, y_std, y_true, num_bins=100, vectorized=False
):
    """Return root mean squared calibration error."""

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins
        )

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce


def mean_absolute_calibration_error(
    y_pred, y_std, y_true, num_bins=100, vectorized=False
):
    """ Return mean absolute calibration error; identical to ECE. """

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins
        )

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace


def adversarial_group_calibration(
    y_pred,
    y_std,
    y_true,
    cali_type,
    num_bins=100,
    num_group_bins=10,
    draw_with_replacement=True,
    num_trials=10,
    num_group_draws=10,
):
    # Flatten
    num_pts = y_true.shape[0]
    y_pred = y_pred.reshape(
        num_pts,
    )
    y_std = y_std.reshape(
        num_pts,
    )
    y_true = y_true.reshape(
        num_pts,
    )

    if cali_type == "mean_abs":
        cali_fn = mean_absolute_calibration_error
    elif cali_type == "root_mean_sq":
        cali_fn = root_mean_squared_calibration_error

    num_pts = y_std.shape[0]
    ratio_arr = np.linspace(0, 1, num_group_bins)
    score_per_ratio = []
    print(
        "Measuring adversarial group calibration by spanning group size between {} and {}, in {} intervals".format(
            np.min(ratio_arr), np.max(ratio_arr), num_group_bins
        )
    )
    for r in tqdm.tqdm(ratio_arr):
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
                )
                group_miscal_scores.append(group_miscal)
            max_miscal_score = np.max(group_miscal_scores)
            score_per_trial.append(max_miscal_score)
        mean_score_across_trials = np.mean(score_per_trial)
        score_per_ratio.append(mean_score_across_trials)

    return ratio_arr, np.array(score_per_ratio)


def miscalibration_area(y_pred, y_std, y_true, num_bins=100, vectorized=False):
    """
    Return miscalibration area.

    This is identical to mean absolute calibration error and ECE, however
    the integration here is taken by tracing the area between curves.
    In the limit of num_bins, miscalibration area and
    mean absolute calibration error will converge to the same value.
    """

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins
        )

    # Compute approximation to area between curves
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy
    ls = LineString(np.c_[x, y])
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    return miscalibration_area


def get_proportion_lists_vectorized(y_pred, y_std, y_true, num_bins=100):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)

    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - exp_proportions / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + exp_proportions / 2.0)
    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound

    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions


def get_proportion_lists(y_pred, y_std, y_true, num_bins=100):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    obs_proportions = [
        get_proportion_in_interval(y_pred, y_std, y_true, quantile)
        for quantile in exp_proportions
    ]

    return exp_proportions, obs_proportions


def get_proportion_in_interval(y_pred, y_std, y_true, quantile):
    """
    For a specified quantile, return the proportion of points falling into
    an interval corresponding to that quantile.
    """

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    lower_bound = norm.ppf(0.5 - quantile / 2)
    upper_bound = norm.ppf(0.5 + quantile / 2)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true
    normalized_residuals = residuals.reshape(-1) / y_std.reshape(-1)
    num_within_quantile = 0
    for resid in normalized_residuals:
        if lower_bound <= resid <= upper_bound:
            num_within_quantile += 1.0
    proportion = num_within_quantile / len(residuals)

    return proportion
