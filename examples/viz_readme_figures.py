"""
Examples of code for visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
import uncertainty_toolbox.viz as uviz

import neatplot

neatplot.set_style()
neatplot.update_rc("text.usetex", True)  # Set to True for system latex


# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
n_obs = 650
f, std, y, x = udata.synthetic_sine_heteroscedastic(n_obs)

# Save figure (set to True to save)
savefig = True


def update_rc_params():
    """Update matplotlib rc params."""
    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"xtick.labelsize": 14})
    plt.rcParams.update({"ytick.labelsize": 14})


def make_plots(pred_mean, pred_std, plot_save_str='row'):
    """Make set of plots."""

    update_rc_params()
    ylims = [-3, 3]
    n_subset = 50

    fig, axs = plt.subplots(1, 3, figsize=(17, 8))

    # Make xy plot
    axs[0] = uviz.plot_xy_ax(
        pred_mean, pred_std, y, x, n_subset=300, ylims=ylims, xlims=[0, 15], ax=axs[0]
    )

    # Make ordered intervals plot
    axs[1] = uviz.plot_intervals_ordered_ax(
        pred_mean, pred_std, y, n_subset=n_subset, ylims=ylims, ax=axs[1]
    )

    # Make calibration plot
    axs[2] = uviz.plot_calibration_ax(pred_mean, pred_std, y, ax=axs[2])

    # Adjust subplots spacing
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None
    )

    # Save figure
    if savefig:
        neatplot.save_figure(plot_save_str, ['svg'], white_background=False)


# List of predictive means and standard deviations
pred_mean_list = [f]

pred_std_list = [
    std * 0.5,  # overconfident
    std * 2.0,  # underconfident
    std,  # correct
]

# Loop through, make plots, and compute metrics
idx_counter = 0
for i, pred_mean in enumerate(pred_mean_list):
    for j, pred_std in enumerate(pred_std_list):
        mace = umetrics.mean_absolute_calibration_error(pred_mean, pred_std, y)
        rmsce = umetrics.root_mean_squared_calibration_error(pred_mean, pred_std, y)
        ma = umetrics.miscalibration_area(pred_mean, pred_std, y)

        idx_counter += 1
        make_plots(pred_mean, pred_std, f'row_{idx_counter}')

        print(f"MACE: {mace}, RMSCE: {rmsce}, MA: {ma}")
