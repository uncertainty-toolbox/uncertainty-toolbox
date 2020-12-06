"""
Proper Scoring Rules for assessing the quality of predictive
uncertainty quantification.
"""

import numpy as np
from scipy import stats


def nll_gaussian(y_pred, y_std, y_true, scaled=True):
    """
    Return negative log likelihood for held out data (y_true) given predictive
    uncertainty with mean (y_pred) and standard-deviation (y_std).
    """

    # Set residuals
    residuals = y_pred - y_true

    # Flatten
    num_pts = y_true.shape[0]
    residuals = residuals.reshape(
        num_pts,
    )
    y_std = y_std.reshape(
        num_pts,
    )

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
    and standard-deviation (y_std). Each test point is given equal weight
    in the overall score over the test set.

    Negatively oriented means a smaller value is more desirable.
    """

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

    # Compute crps
    y_standardized = (y_true - y_pred) / y_std
    term_1 = 1 / np.sqrt(np.pi)
    term_2 = 2 * stats.norm.pdf(y_standardized, loc=0, scale=1)
    term_3 = y_standardized * (2 * stats.norm.cdf(y_standardized, loc=0, scale=1) - 1)

    crps_list = -1 * y_std * (term_1 - term_2 - term_3)
    crps = np.sum(crps_list)

    # Potentially scale so that sum becomes mean
    if scaled:
        crps = crps / len(crps_list)

    return crps


def check_score(
    y_pred, y_std, y_true, scaled=True, start_q=0.01, end_q=0.99, resolution=99
):
    """
    Return the negatively oriented check score for held out data (y_true)
    given predictive uncertainty with mean (y_pred) and
    standard-deviation (y_std).
    Each test point and each quantile is given equal weight
    in the overall score over the test set and list of quantiles.

    Negatively oriented means a smaller value is more desirable.
    """
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

    test_qs = np.linspace(start_q, end_q, resolution)

    check_list = []
    for q in test_qs:
        q_level = stats.norm.ppf(q, loc=y_pred, scale=y_std)  # pred quantile
        diff = q_level - y_true
        mask = (diff >= 0).astype(float) - q
        score_per_q = np.mean(mask * diff)
        check_list.append(score_per_q)
    check_score = np.sum(check_list)

    if scaled:
        check_score = check_score / len(check_list)

    return check_score


def interval_score(
    y_pred, y_std, y_true, scaled=True, start_p=0.01, end_p=0.99, resolution=99
):
    """
    Return the negatively oriented interval score for held out data (y_true)
    given predictive uncertainty with mean (y_pred) and standard-deviation
    (y_std). Each test point and each percentile is given equal weight in the
    overall score over the test set and list of quantiles.

    Negatively oriented means a smaller value is more desirable.
    """
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

    test_ps = np.linspace(start_p, end_p, resolution)

    int_list = []
    for p in test_ps:
        low_p, high_p = 0.5 - (p / 2.0), 0.5 + (p / 2.0)  # p% PI
        pred_l = stats.norm.ppf(low_p, loc=y_pred, scale=y_std)
        pred_u = stats.norm.ppf(high_p, loc=y_pred, scale=y_std)

        below_l = ((pred_l - y_true) > 0).astype(float)
        above_u = ((y_true - pred_u) > 0).astype(float)

        score_per_p = (
            (pred_u - pred_l)
            + (2.0 / (1 - p)) * (pred_l - y_true) * below_l
            + (2.0 / (1 - p)) * (y_true - pred_u) * above_u
        )
        mean_score_per_p = np.mean(score_per_p)
        int_list.append(mean_score_per_p)
    int_score = np.sum(int_list)

    if scaled:
        int_score = int_score / len(int_list)

    return int_score
