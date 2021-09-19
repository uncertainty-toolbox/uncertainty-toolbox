from argparse import Namespace
import numpy as np
import torch
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from tqdm import tqdm

from uncertainty_toolbox.metrics_accuracy import prediction_error_metrics


""" Utilities """
def get_quantile_model_predictions(
    model,
    x,
    exp_proportions,
    recal_model=None,
    recal_type=None,
):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(model.device)
    if isinstance(exp_proportions, np.ndarray):
        exp_proportions = torch.from_numpy(exp_proportions).to(model.device)

    assert isinstance(x, torch.Tensor)
    assert isinstance(exp_proportions, torch.Tensor)

    quantile_preds = model.forward(
        x=x,
        q_list=exp_proportions,
        recal_model=recal_model,
        recal_type=recal_type,
    )  # of shape (num_x, num_q)
    quantile_preds_arr = quantile_preds.detach().cpu().numpy()

    return quantile_preds_arr


""" Accuracy """


def quantile_accuracy(
    model,
    x,
    y,
    recal_model=None,
    recal_type=None
):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy().flatten()

    median_quantile = np.array([0.5])
    quantile_predictions = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=median_quantile,
        recal_model=recal_model,
        recal_type=recal_type,
    )
    q_050 = quantile_predictions[:, 0]

    return prediction_error_metrics(y_pred=q_050, y_true=y)


""" Sharpness """


def quantile_sharpness(
    model,
    x,
    recal_model=None,
    recal_type=None,
):
    exp_proportions = np.array([0.025, 0.975])
    quantile_predictions = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=exp_proportions,
        recal_model=recal_model,
        recal_type=recal_type,
    )
    q_025 = quantile_predictions[:, 0]
    q_975 = quantile_predictions[:, 1]
    sharp_metric = np.mean(np.abs(q_975 - q_025))

    return sharp_metric


""" Calibration """


def quantile_root_mean_squared_calibration_error(
    model,
    x,
    y,
    num_bins=99,
    recal_model=None,
    recal_type=None,
    prop_type="quantile",
):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    exp_proportions = np.linspace(0.01, 0.99, num_bins)
    quantile_predictions = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=exp_proportions,
        recal_model=recal_model,
        recal_type=recal_type,
    )
    obs_proportions = np.mean((quantile_predictions >= y).astype(float), axis=0).flatten()
    assert exp_proportions.shape == obs_proportions.shape

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce


def quantile_mean_absolute_calibration_error(
    model,
    x,  # shape (N, 1)
    y,  # shape (N, 1)
    num_bins=99,
    recal_model=None,
    recal_type=None,
    prop_type="quantile",
):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()


    exp_proportions = np.linspace(0.01, 0.99, num_bins)
    quantile_predictions = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=exp_proportions,
        recal_model=recal_model,
        recal_type=recal_type,
    )
    obs_proportions = np.mean((quantile_predictions >= y).astype(float), axis=0).flatten()
    assert exp_proportions.shape == obs_proportions.shape

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace


def quantile_miscalibration_area(
    model,
    x,
    y,
    num_bins=99,
    recal_model=None,
    recal_type=None,
    prop_type="quantile",
):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    exp_proportions = np.linspace(0.01, 0.99, num_bins)
    quantile_predictions = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=exp_proportions,
        recal_model=recal_model,
        recal_type=recal_type,
    )
    obs_proportions = np.mean((quantile_predictions >= y).astype(float), axis=0).flatten()
    assert exp_proportions.shape == obs_proportions.shape

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


def quantile_adversarial_group_calibration(
    model,
    x,
    y,
    cali_type='mean_abs',
    prop_type="quantile",
    num_bins=99,
    num_group_bins=10,
    draw_with_replacement=False,
    num_trials=10,
    num_group_draws=10,
    verbose=False,
):

    if cali_type == "mean_abs":
        cali_fn = quantile_mean_absolute_calibration_error
    elif cali_type == "root_mean_sq":
        cali_fn = quantile_root_mean_squared_calibration_error

    num_pts = y.shape[0]
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
                group_x = x[rand_idx]
                group_y = y[rand_idx]
                group_miscal = cali_fn(
                    model,
                    group_x,
                    group_y,
                    num_bins=num_bins,
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


""" Proper scoring rules """
def quantile_check_score(
    model,
    x,
    y,
    num_bins=99,
    recal_model=None,
    recal_type=None,
):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    q_list = np.linspace(0.01, 0.99, num_bins)

    quantile_predictions = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=q_list,
        recal_model=recal_model,
        recal_type=recal_type,
    )

    diff = quantile_predictions - y
    mask = (diff >= 0).astype(float) - q_list
    check_score = np.mean((mask * diff))

    return check_score


def quantile_interval_score(
    model,
    x,
    y,
    num_bins=99,
    recal_model=None,
    recal_type=None,
):

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    num_pts = y.shape[0]

    alpha_list = np.linspace(0.01, 0.99, num_bins)
    l_list = np.min(
        np.stack([(alpha_list / 2.0), 1 - (alpha_list / 2.0)], axis=1),
        axis=1
    )
    u_list = 1.0 - l_list

    pred_l = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=l_list,
        recal_model=recal_model,
        recal_type=recal_type,
    )
    pred_u = get_quantile_model_predictions(
        model=model,
        x=x,
        exp_proportions=u_list,
        recal_model=recal_model,
        recal_type=recal_type,
    )

    below_l = (pred_l - y) > 0
    above_u = (y - pred_u) > 0

    score_per_alpha = (
            (pred_u - pred_l)
            + (1.0 / l_list) * (pred_l - y) * below_l
            + (1.0 / l_list) * (y - pred_u) * above_u
    )

    int_score = np.mean(score_per_alpha)

    return int_score

