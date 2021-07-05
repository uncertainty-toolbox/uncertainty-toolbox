"""
Recalibrating uncertainty estimates.
"""
import tqdm
import numpy as np
from sklearn.isotonic import IsotonicRegression
import uncertainty_toolbox as uct
from scipy import special
from scipy.optimize import minimize_scalar


def get_q_idx(exp_props, q):
    num_pts = exp_props.shape[0]
    target_idx = None
    for idx, x in enumerate(exp_props):
        if idx + 1 == num_pts:
            if round(q, 2) == round(float(exp_props[-1]), 2):
                target_idx = exp_props.shape[0] - 1
            break
        if x <= q < exp_props[idx + 1]:
            target_idx = idx
            break
    if target_idx is None:
        raise ValueError("q must be within exp_props")
    return target_idx


def iso_recal(exp_props, obs_props):
    """
    Returns an isotonic regression model that maps from obs_props to exp_props
    """
    # Flatten
    exp_props = exp_props.flatten()
    obs_props = obs_props.flatten()
    min_obs = np.min(obs_props)
    max_obs = np.max(obs_props)

    iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    # just need observed prop values between 0 and 1
    # problematic if min_obs_p > 0 and max_obs_p < 1
    if not (min_obs == 0.0) and (max_obs == 1.0):
        print("Obs props not ideal: from {} to {}".format(min_obs, max_obs))

    exp_0_idx = get_q_idx(exp_props, 0.0)
    exp_1_idx = get_q_idx(exp_props, 1.0)
    within_01 = obs_props[exp_0_idx : exp_1_idx + 1]

    beg_idx, end_idx = None, None
    # Handle beg_idx
    if exp_0_idx != 0:
        min_obs_below = np.min(obs_props[:exp_0_idx])
        min_obs_within = np.min(within_01)
        if min_obs_below < min_obs_within:
            i = exp_0_idx - 1
            while obs_props[i] > min_obs_below:
                i -= 1
            beg_idx = i
        elif np.sum((within_01 == min_obs_within).astype(float)) > 1:
            # multiple minima in within_01 ==> get last min idx
            i = exp_1_idx - 1
            while obs_props[i] > min_obs_within:
                i -= 1
            beg_idx = i
        elif np.sum((within_01 == min_obs_within).astype(float)) == 1:
            beg_idx = int(np.argmin(within_01) + exp_0_idx)
        else:
            raise RuntimeError(
                ("Inspect input arrays, " "cannot set beginning index.")
            )
    else:
        beg_idx = exp_0_idx

    # Handle end_idx
    if exp_1_idx < obs_props.shape[0] - 1:
        max_obs_above = np.max(obs_props[exp_1_idx + 1 :])
        max_obs_within = np.max(within_01)
        if max_obs_above > max_obs_within:
            i = exp_1_idx + 1
            while obs_props[i] < max_obs_above:
                i += 1
            end_idx = i + 1
        elif np.sum((within_01 == max_obs_within).astype(float)) > 1:
            # multiple minima in within_01 ==> get last min idx
            i = beg_idx
            while obs_props[i] < max_obs_within:
                i += 1
            end_idx = i + 1
        elif np.sum((within_01 == max_obs_within).astype(float)) == 1:
            end_idx = int(exp_0_idx + np.argmax(within_01) + 1)
        else:
            raise RuntimeError("Inspect input arrays, cannot set ending index")
    else:
        end_idx = exp_1_idx + 1

    if end_idx <= beg_idx:
        raise RuntimeError("Ending index before beginning index")

    filtered_obs_props = obs_props[beg_idx:end_idx]
    filtered_exp_props = exp_props[beg_idx:end_idx]

    try:
        iso_model = iso_model.fit(filtered_obs_props, filtered_exp_props)
    except Exception:
        raise RuntimeError("Failed to fit isotonic regression model")

    return iso_model


def optimize_recalibration_ratio(y_mean, y_std, y_true, criterion="ma_cal"):
    if criterion == "ma_cal":
        cal_fn = uct.metrics.mean_absolute_calibration_error
        worst_cal = 0.5
    elif criterion == "rms_cal":
        cal_fn = uct.metrics.root_mean_squared_calibration_error
        worst_cal = np.sqrt(1.0/3.0)
    elif criterion == "miscal":
        cal_fn = uct.metrics.miscalibration_area
        worst_cal = 0.5
    else:
        raise RuntimeError("Wrong criterion option")

    def obj(ratio):

        # If ratio is 0, return worst-possible calibration metric
        if ratio == 0:
            return worst_cal

        curr_cal = cal_fn(y_mean, ratio * y_std, y_true)
        return curr_cal

    bounds = (1e-3, 1e3)
    result = minimize_scalar(fun=obj, bounds=bounds)
    opt_ratio = result.x

    if not result.success:
        raise Warning("Optimization did not succeed")
        original_cal = cal_fn(y_mean, y_std, y_true)
        ratio_cal = cal_fn(y_mean, opt_ratio * y_std, y_true)
        if ratio_cal > original_cal:
            raise Warning(
                "No better calibration found, no recalibration performed and returning original uncertainties"
            )
            opt_ratio = 1.0

    return opt_ratio
