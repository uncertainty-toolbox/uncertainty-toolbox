"""
Examples of code for visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct


# Set plot style
uct.viz.set_style()
uct.viz.update_rc("text.usetex", True)  # Set to True for system latex
uct.viz.update_rc("font.size", 14)  # Set font size
uct.viz.update_rc("xtick.labelsize", 14)  # Set font size for xaxis tick labels
uct.viz.update_rc("ytick.labelsize", 14)  # Set font size for yaxis tick labels

# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
n_obs = 650
f, std, y, x = uct.synthetic_sine_heteroscedastic(n_obs)

# Save figure (set to True to save)
savefig = True


def make_plots(pred_mean, pred_std, plot_save_str="row"):
    """Make set of plots."""

    ylims = [-3, 3]
    n_subset = 50

    fig, axs = plt.subplots(1, 3, figsize=(17, 8))

    # Make xy plot
    axs[0] = uct.plot_xy(
        pred_mean, pred_std, y, x, n_subset=300, ylims=ylims, xlims=[0, 15], ax=axs[0]
    )

    # Make ordered intervals plot
    axs[1] = uct.plot_intervals_ordered(
        pred_mean, pred_std, y, n_subset=n_subset, ylims=ylims, ax=axs[1]
    )

    # Make calibration plot
    axs[2] = uct.plot_calibration(pred_mean, pred_std, y, ax=axs[2])

    # Adjust subplots spacing
    fig.subplots_adjust(wspace=0.25)

    # Save figure
    if savefig:
        uct.viz.save_figure(plot_save_str, "svg", white_background=True)


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
        mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y)
        rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y)
        ma = uct.miscalibration_area(pred_mean, pred_std, y)

        idx_counter += 1
        make_plots(pred_mean, pred_std, f"row_{idx_counter}")

        print(f"MACE: {mace}, RMSCE: {rmsce}, MA: {ma}")
