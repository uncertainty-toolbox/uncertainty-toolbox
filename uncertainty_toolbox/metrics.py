"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""

import os, sys
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from metrics_accuracy import prediction_error_metrics
from metrics_calibration import (
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    miscalibration_area,
    adversarial_group_calibration,
    sharpness,
)
from metrics_scoring_rule import (
    nll_gaussian,
    crps_gaussian,
    check_score,
    interval_score,
)


def parse_run_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cali_bins",
        type=int,
        default=100,
        help="number of bins to discretize probabilities for calibration",
    )
    parser.add_argument(
        "--sr_bins",
        type=int,
        default=99,
        help="number of bins to discretize probabilities for scoring rules",
    )
    parser.add_argument(
        "--sr_scale", type=int, default=1, help="1 to scale scoring rules outputs"
    )
    # parser.add_argument('--', type=, default=, help='')

    options = parser.parse_args()
    options.sr_scale = bool(options.sr_scale)

    return options


def get_all_metrics(y_pred, y_std, y_true):
    options = parse_run_options()

    """ Accuracy """
    print("Calculating accuracy metrics...")
    acc_metrics = prediction_error_metrics(y_pred, y_true)

    """ Calibration """
    print("Calculating average calibration metrics...")
    cali_metrics = {}
    cali_metrics["rms_cali"] = root_mean_squared_calibration_error(
        y_pred, y_std, y_true, num_bins=options.cali_bins
    )
    cali_metrics["ma_cali"] = mean_absolute_calibration_error(
        y_pred, y_std, y_true, num_bins=options.cali_bins
    )
    cali_metrics["miscal_area"] = miscalibration_area(
        y_pred, y_std, y_true, num_bins=options.cali_bins
    )

    """ Adversarial Group Calibration """
    print("Calculating adversarial group calibration metrics...")
    print("...for mean absolute calibration error")
    ma_adv_group_size, ma_adv_group_cali = adversarial_group_calibration(
        y_pred,
        y_std,
        y_true,
        cali_type="mean_abs",
        num_bins=options.cali_bins,
    )
    cali_metrics["ma_adv_group_cali"] = {
        "group_sizes": ma_adv_group_size,
        "adv_group_cali": ma_adv_group_cali,
    }

    print("...for root mean squared calibration error")
    rms_adv_group_size, rms_adv_group_cali = adversarial_group_calibration(
        y_pred,
        y_std,
        y_true,
        cali_type="mean_abs",
        num_bins=options.cali_bins,
    )
    cali_metrics["rms_adv_group_cali"] = {
        "group_sizes": rms_adv_group_size,
        "adv_group_cali": rms_adv_group_cali,
    }

    """ Sharpness """
    print("Calculating sharpness metrics...")
    cali_metrics["sharp"] = sharpness(y_std)

    """ Proper Scoring Rules """
    print("Calculating proper scoring rule metrics...")
    sr_metrics = {}
    sr_metrics["nll"] = nll_gaussian(y_pred, y_std, y_true, scaled=options.sr_scale)
    sr_metrics["crps"] = crps_gaussian(y_pred, y_std, y_true, scaled=options.sr_scale)
    sr_metrics["check"] = check_score(
        y_pred, y_std, y_true, scaled=options.sr_scale, resolution=options.sr_bins
    )
    sr_metrics["interval"] = interval_score(
        y_pred, y_std, y_true, scaled=options.sr_scale, resolution=options.sr_bins
    )

    # Print all outputs
    for acc_metric, acc_val in acc_metrics.items():
        print("{:}: {:.3f}".format(acc_metric, acc_val))
    for cali_metric, cali_val in cali_metrics.items():
        if "adv_group_cali" not in cali_metric:
            print("{:}: {:.3f}".format(cali_metric, cali_val))
        else:
            print("{:}:".format(cali_metric))
            for subitem, subval in cali_val.items():
                print("{:>15}: {:}".format(subitem, np.around(subval, decimals=3)))
    for sr_metric, sr_val in sr_metrics.items():
        print("{:}: {:.3f}".format(sr_metric, sr_val))

    all_scores = {
        "accuracy": acc_metrics,
        "cali_sharp": cali_metrics,
        "scoring_rule": sr_metrics,
    }

    return all_scores


if __name__ == "__main__":
    y_pred = np.array([1, 2, 3, 4])
    y_std = np.array([1, 2, 3, 4])
    y_true = np.array([1.3, 2.3, 3.3, 4])
    get_all_metrics(y_pred, y_std, y_true)
