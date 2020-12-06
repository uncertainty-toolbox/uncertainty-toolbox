"""
Examples of code for recalibration.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists_vectorized,
)
import uncertainty_toolbox.viz as uviz
from uncertainty_toolbox.recalibration import iso_recal
from uncertainty_toolbox.viz import plot_calibration

import neatplot

neatplot.set_style()
neatplot.update_rc("text.usetex", False)

# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
n_obs = 650
f, std, y, x = udata.synthetic_sine_heteroscedastic(n_obs)

# Save figure (set to True to save)
savefig = False


def save_figure(name_str, file_type="png"):
    """Save figure, or do nothing if savefig is False."""
    if savefig:
        neatplot.save_figure(name_str, file_type)


def update_rc_params():
    """Update matplotlib rc params."""
    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"xtick.labelsize": 14})
    plt.rcParams.update({"ytick.labelsize": 14})


# List of predictive means and standard deviations
pred_mean_list = [
    f,
    # f + 0.1,
    # f - 0.1,
    # f + 0.25,
    # f - 0.25,
]

pred_std_list = [
    std * 0.5,  # overconfident
    std * 2.0,  # underconfident
    # std,                # correct
]

# Loop through, make plots, and compute metrics
for i, pred_mean in enumerate(pred_mean_list):
    for j, pred_std in enumerate(pred_std_list):
        # Before recalibration
        exp_props, obs_props = get_proportion_lists_vectorized(
            pred_mean, pred_std, y
        )
        recal_model = None
        mace = umetrics.mean_absolute_calibration_error(
            pred_mean, pred_std, y, recal_model=recal_model
        )
        rmsce = umetrics.root_mean_squared_calibration_error(
            pred_mean, pred_std, y, recal_model=recal_model
        )
        ma = umetrics.miscalibration_area(
            pred_mean, pred_std, y, recal_model=recal_model
        )
        print("Before Recalibration")
        print(
            "   MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma)
        )

        uviz.plot_calibration(
            pred_mean,
            pred_std,
            y,
            exp_props=exp_props,
            obs_props=obs_props,
            show=True,
        )

        # After recalibration
        recal_model = iso_recal(exp_props, obs_props)
        recal_exp_props, recal_obs_props = get_proportion_lists_vectorized(
            pred_mean, pred_std, y, recal_model=recal_model
        )
        mace = umetrics.mean_absolute_calibration_error(
            pred_mean, pred_std, y, recal_model=recal_model
        )
        rmsce = umetrics.root_mean_squared_calibration_error(
            pred_mean, pred_std, y, recal_model=recal_model
        )
        ma = umetrics.miscalibration_area(
            pred_mean, pred_std, y, recal_model=recal_model
        )
        print(" After Recalibration")
        print(
            "   MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma)
        )

        plot_calibration(
            pred_mean,
            pred_std,
            y,
            exp_props=recal_exp_props,
            obs_props=recal_obs_props,
            show=True,
        )
