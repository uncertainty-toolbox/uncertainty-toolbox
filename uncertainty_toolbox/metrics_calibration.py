"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""

from argparse import Namespace
import numpy as np
from scipy import stats
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from tqdm import tqdm
from uncertainty_toolbox.utils import assert_is_flat_same_shape, assert_is_positive


def sharpness(y_std):
    """
    Return sharpness (a single measure of the overall confidence).
    """
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_std)
    # Check that input std is positive
    assert_is_positive(y_std)

    # Compute sharpness
    sharp_metric = np.sqrt(np.mean(y_std ** 2))

    return sharp_metric


def root_mean_squared_calibration_error(
    y_pred, y_std, y_true, num_bins=100, vectorized=False, recal_model=None, prop_type='interval'
):
    """Return root mean squared calibration error."""

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ['interval', 'quantile']

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
    y_pred, y_std, y_true, num_bins=100, vectorized=False, recal_model=None, prop_type='interval'
):
    """Return mean absolute calibration error; identical to ECE."""

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ['interval', 'quantile']

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
    y_pred,
    y_std,
    y_true,
    cali_type,
    prop_type='interval',
    num_bins=100,
    num_group_bins=10,
    draw_with_replacement=False,
    num_trials=10,
    num_group_draws=10,
    verbose=False,
):

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ['interval', 'quantile']

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
    y_pred, y_std, y_true, num_bins=100, vectorized=False, recal_model=None, prop_type='interval'
):
    """
    Return miscalibration area.

    This is identical to mean absolute calibration error and ECE, however
    the integration here is taken by tracing the area between curves.
    In the limit of num_bins, miscalibration area and
    mean absolute calibration error will converge to the same value.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ['interval', 'quantile']

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type=prop_type
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type=prop_type
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


def get_proportion_lists_vectorized(
    y_pred, y_std, y_true, num_bins=100, recal_model=None, prop_type='interval'
):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ['interval', 'quantile']

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(
        -1, 1
    )
    norm = stats.norm(loc=0, scale=1)
    if prop_type == 'interval':
        gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)
        gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)

        above_lower = normalized_residuals >= gaussian_lower_bound
        below_upper = normalized_residuals <= gaussian_upper_bound

        within_quantile = above_lower * below_upper
        obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)
    elif prop_type == 'quantile':
        gaussian_quantile_bound = norm.ppf(in_exp_proportions)
        below_quantile = normalized_residuals <= gaussian_quantile_bound
        obs_proportions = np.sum(below_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions


def get_proportion_lists(y_pred, y_std, y_true, num_bins=100, recal_model=None, prop_type='interval'):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ['interval', 'quantile']

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    if prop_type == 'interval':
        obs_proportions = [
            get_proportion_in_interval(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]
    elif prop_type == 'quantile':
        obs_proportions = [
            get_proportion_under_quantile(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]

    return exp_proportions, obs_proportions


def get_proportion_in_interval(y_pred, y_std, y_true, quantile):
    """
    For a specified quantile, return the proportion of points falling into
    an interval corresponding to that quantile.
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


def get_proportion_under_quantile(y_pred, y_std, y_true, quantile):
    """
    For a specified quantile, return the proportion of points falling under
    a predicted quantile.
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


def get_prediction_interval(y_pred, y_std, quantile, recal_model=None):
    """
    For a specified quantile level q (must be a float, or a singleton),
    return the centered prediction interval corresponding
    to the pair of quantiles at levels (0.5-q/2) and (0.5+q/2),
    i.e. interval that has nominal coverage equal to q.
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


def get_quantile(y_pred, y_std, quantile, recal_model=None):
    """
    For a specified quantile level q (must be a float, or a singleton),
    return the quantile prediction,
    i.e. upper bound that has nominal coverage below the bound equal to q.
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
