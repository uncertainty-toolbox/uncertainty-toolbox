import numpy as np
import torch

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


def quantile_root_mean_squared_calibration_error(
    model,
    x,
    y,
    num_bins=99,
    recal_model=None,
    recal_type=None,
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
    x,
    y,
    num_bins=99,
    recal_model=None,
    recal_type=None,
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
    sharp_metric = np.mean(q_975 - q_025)

    return sharp_metric


# def test_uq(
#     model,
#     x,
#     y,
#     exp_props,
#     y_range,
#     recal_model=None,
#     recal_type=None,
#     make_plots=False,
#     test_group_cal=False,
# ):
#     g_cali_scores = []
#     if test_group_cal:
#         ratio_arr = np.linspace(0.01, 1.0, 10)
#         print(
#             "Spanning group size from {} to {} in {} increments".format(
#                 np.min(ratio_arr), np.max(ratio_arr), len(ratio_arr)
#             )
#         )
#         for r in tqdm.tqdm(ratio_arr):
#             gc = test_group_cali(
#                 y=y,
#                 q_pred_mat=quantile_preds[:, idx_01 : idx_99 + 1],
#                 exp_props=exp_props[idx_01 : idx_99 + 1],
#                 y_range=y_range,
#                 ratio=r,
#             )
#             g_cali_scores.append(gc)
#         g_cali_scores = np.array(g_cali_scores)
#
#         # plt.plot(ratio_arr, g_cali_scores)
#         # plt.show()
#
#     """ Get some scoring rules """
#     args = Namespace(
#         device=torch.device("cpu"), q_list=torch.linspace(0.01, 0.99, 99)
#     )
#     curr_check = check_loss(model, x, y, args)
#     curr_int = interval_score(model, x, y, args)
#     int_exp_props, int_obs_props, int_cdf_preds = get_obs_props(
#         model, x, y, exp_props=None, device=args.device, type="interval"
#     )
#     curr_int_cali = torch.mean(torch.abs(int_exp_props - int_obs_props)).item()
#
#     curr_scoring_rules = {
#         "check": float(curr_check),
#         "int": float(curr_int),
#         "int_cali": float(curr_int_cali),
#     }
#
#     return (
#         obs_props,
#         quantile_preds,
#         g_cali_scores,
#         curr_scoring_rules,
#     )
#
#
# def test_group_cali(
#     y,
#     q_pred_mat,
#     exp_props,
#     y_range,
#     ratio,
#     num_group_draws=20,
#     make_plots=False,
# ):
#
#     num_pts, num_q = q_pred_mat.shape
#     group_size = max([int(round(num_pts * ratio)), 2])
#     q_025_idx = get_q_idx(exp_props, 0.025)
#     q_975_idx = get_q_idx(exp_props, 0.975)
#
#     # group_obs_props = []
#     # group_sharp_scores = []
#     score_per_trial = []
#     for _ in range(20):  # each trial
#         ##########
#         group_cali_scores = []
#         for g_idx in range(num_group_draws):
#             rand_idx = np.random.choice(num_pts, group_size, replace=True)
#             g_y = y[rand_idx]
#             g_q_preds = q_pred_mat[rand_idx, :]
#             g_obs_props = torch.mean(
#                 (g_q_preds >= g_y).float(), dim=0
#             ).flatten()
#             assert exp_props.shape == g_obs_props.shape
#             g_cali_score = plot_calibration_curve(
#                 exp_props, g_obs_props, make_plots=False
#             )
#             # g_sharp_score = torch.mean(
#             #     g_q_preds[:,q_975_idx] - g_q_preds[:,q_025_idx]
#             # ).item() / y_range
#
#             group_cali_scores.append(g_cali_score)
#             # group_obs_props.append(g_obs_props)
#             # group_sharp_scores.append(g_sharp_score)
#
#         # mean_cali_score = np.mean(group_cali_scores)
#         mean_cali_score = np.max(group_cali_scores)
#         ##########
#
#         score_per_trial.append(mean_cali_score)
#
#     return np.mean(score_per_trial)
#
#     # mean_sharp_score = np.mean(group_sharp_scores)
#     # mean_group_obs_props = torch.mean(torch.stack(group_obs_props, dim=0), dim=0)
#     # mean_group_cali_score = plot_calibration_curve(exp_props, mean_group_obs_props,
#     #                                                make_plots=False)
#
#     return mean_cali_score
