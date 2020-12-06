"""
Examples of code for recalibration.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
from uncertainty_toolbox.metrics_calibration import get_proportion_lists_vectorized
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


def make_plots(pred_mean, pred_std, idx1, idx2):
    """Make set of plots."""

    update_rc_params()
    ylims = [-3, 3]
    n_subset = 50

    # Make xy plot
    uviz.plot_xy(pred_mean, pred_std, y, x, n_subset=300, ylims=ylims, xlims=[0, 15])
    save_figure(f"xy_{idx1}_{idx2}")
    plt.show()

    # Make intervals plot
    uviz.plot_intervals(pred_mean, pred_std, y, n_subset=n_subset, ylims=ylims)
    save_figure(f"intervals_{idx1}_{idx2}")
    plt.show()

    # Make calibration plot
    uviz.plot_calibration(pred_mean, pred_std, y)
    save_figure(f"calibration_{idx1}_{idx2}")
    plt.show()

    # Make ordered intervals plot
    uviz.plot_intervals_ordered(pred_mean, pred_std, y, n_subset=n_subset, ylims=ylims)
    save_figure(f"intervals_ordered_{idx1}_{idx2}")
    plt.show()


# List of predictive means and standard deviations
pred_mean_list = [f, f + 0.1]

pred_std_list = [
    std * 0.5,  # overconfident
    std * 2.0,  # underconfident
]

# Loop through, make plots, and compute metrics
for i, pred_mean in enumerate(pred_mean_list):
    for j, pred_std in enumerate(pred_std_list):
        # Before recalibration
        exp_props, obs_props = get_proportion_lists_vectorized(pred_mean, pred_std, y)
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
        print("   MACE: {:.3f}, RMSCE: {:.3f}, MA: {:.3f}".format(mace, rmsce, ma))

        plot_calibration(
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
        print("   MACE: {:.3f}, RMSCE: {:.3f}, MA: {:.3f}".format(mace, rmsce, ma))

        plot_calibration(
            pred_mean,
            pred_std,
            y,
            exp_props=recal_exp_props,
            obs_props=recal_obs_props,
            show=True,
        )
