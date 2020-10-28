"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             median_absolute_error)
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

""" Proper Scoring Rules """
def nll_gaussian(y_pred, y_std, y_true, scaled=True):
    """
    Return negative log likelihood for held out data (y_true) given predictive
    uncertainty with mean (y_pred) and standard-deviation (y_std).
    """

    # Set residuals
    residuals = y_pred - y_true

    # Flatten
    num_pts = y_true.shape[0]
    residuals = residuals.reshape(num_pts,)
    y_std = y_std.reshape(num_pts,)

    # Compute nll
    nll_list = stats.norm.logpdf(residuals, scale=y_std)
    nll = -1 * np.sum(nll_list)

    # Potentially scale so that sum becomes mean
    if scaled:
      nll = nll / len(nll_list)

    return nll


def crps_gaussian(y_pred, y_std, y_true, scaled=True):
    """
    Return the negatively oriented continuous ranked probability score for
    held out data (y_true) given predictive uncertainty with mean (y_pred)
    and standard-deviation (y_std).
    """

    # Flatten
    num_pts = y_true.shape[0]
    y_pred = y_pred.reshape(num_pts,)
    y_std = y_std.reshape(num_pts,)
    y_true = y_true.reshape(num_pts,)

    # Compute crps
    y_standardized = (y_true - y_pred) / y_std
    term_1 = 1/np.std(np.pi)
    term_2 = 2 * stats.norm.pdf(y_standardized)
    term_3 = y_standardized * (2 * stats.norm.cdf(y_standardized) - 1)
    crps_list = y_std * (term_1 - term_2 - term_3)
    crps = -1 * np.sum(crps_list)

    # Potentially scale so that sum becomes mean
    if scaled:
        crps = crps / len(crps_list)

    return crps


""" Error, Calibration, Sharpness Metrics """

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

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mdae = median_absolute_error(y_true, y_pred)
    marpd = np.abs(2 * residuals / (np.abs(y_pred)
                   + np.abs(y_true))
                  ).mean() * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    prediction_metrics = {'mae':mae, 'rmse':rmse, 'mdae':mdae, 'marpd':marpd,
                          'r2':r2, 'corr':corr}

    return prediction_metrics


def sharpness(y_std):
    """
    Return sharpness (a single measure of the overall confidence).
    """

    # Compute sharpness
    sharpness = np.sqrt(np.mean(y_std**2))

    return sharpness


def root_mean_squared_calibration_error(y_pred, y_std, y_true, num_bins=100):
    """Return root mean squared calibration error."""

    # Get lists of expected and observed proportions for a range of quantiles
    (exp_proportions, obs_proportions) = get_proportion_lists(y_pred, y_std, y_true, num_bins)

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce


def mean_absolute_calibration_error(y_pred, y_std, y_true, num_bins=100):
    """Return mean absolute calibration error; identical to ECE."""

    # Get lists of expected and observed proportions for a range of quantiles
    (exp_proportions, obs_proportions) = get_proportion_lists(y_pred, y_std, y_true, num_bins)

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace


def miscalibration_area(y_pred, y_std, y_true, num_bins=100):
    """Return miscalibration area."""

    # Get lists of expected and observed proportions for a range of quantiles
    # (exp_proportions, obs_proportions) = get_proportion_lists(y_pred, y_std, y_true, num_bins)
    (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(y_pred, y_std, y_true, num_bins)

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
    polygon_area_list =[poly.area for poly in polygonize(mls)]
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
    gaussian_lower_bound = norm.ppf(0.5 - exp_proportions/2.0)
    gaussian_upper_bound = norm.ppf(0.5 + exp_proportions/2.0)
    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1,1)
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
    obs_proportions = [get_proportion_in_interval(y_pred, y_std, y_true, quantile)
                       for quantile in exp_proportions]

    return exp_proportions, obs_proportions


def get_proportion_in_interval(y_pred, y_std, y_true, quantile):
    """
    For a specified quantile, return the proportion of points falling into
    an interval corresponding to that quantile.
    """

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    lower_bound = norm.ppf(0.5 - quantile/2)
    upper_bound = norm.ppf(0.5 + quantile/2)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true
    normalized_residuals = residuals.reshape(-1) / y_std.reshape(-1)
    num_within_quantile = 0
    for resid in normalized_residuals:
        if lower_bound <= resid <= upper_bound:
            num_within_quantile += 1.
    proportion = num_within_quantile / len(residuals)

    return proportion
