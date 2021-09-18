"""
Examples of code for recalibration.
"""

import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct


# Set plot style
uct.viz.set_style()
uct.viz.update_rc("text.usetex", False)  # Set to True for system latex
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

# List of predictive means and standard deviations
pred_mean_list = [f]
pred_std_list = [
    std * 0.5,  # overconfident
    std * 2.0,  # underconfident
]

# ylims for xy plot
ylims_xy = (-2.51, 3.31)

# Loop through, make plots, and compute metrics
for i, pred_mean in enumerate(pred_mean_list):
    for j, pred_std in enumerate(pred_std_list):

        # Before recalibration
        exp_props, obs_props = uct.get_proportion_lists_vectorized(
            pred_mean, pred_std, y
        )
        mace = uct.mean_absolute_calibration_error(
            pred_mean, pred_std, y, recal_model=None
        )
        rmsce = uct.root_mean_squared_calibration_error(
            pred_mean, pred_std, y, recal_model=None
        )
        ma = uct.miscalibration_area(pred_mean, pred_std, y, recal_model=None)
        print("Before Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        uct.plot_calibration(
            pred_mean,
            pred_std,
            y,
            exp_props=exp_props,
            obs_props=obs_props,
            ax=axes.flatten()[0],
        )
        uct.plot_xy(
            pred_mean,
            pred_std,
            y,
            x,
            ax=axes.flatten()[1],
            ylims=ylims_xy,
        )

        uct.viz.save_figure(f"before_recal_{j}", "png")

        # After recalibration
        std_recalibrator = uct.recalibration.get_std_recalibrator(
            pred_mean, pred_std, y
        )
        pred_std_recal = std_recalibrator(pred_std)

        mace = uct.mean_absolute_calibration_error(pred_mean, pred_std_recal, y)
        rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std_recal, y)
        ma = uct.miscalibration_area(pred_mean, pred_std_recal, y)
        print("After Recalibration:  ", end="")
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        uct.plot_calibration(
            pred_mean,
            pred_std_recal,
            y,
            ax=axes.flatten()[0],
        )
        uct.plot_xy(
            pred_mean,
            pred_std_recal,
            y,
            x,
            ax=axes.flatten()[1],
            ylims=ylims_xy,
        )

        uct.viz.save_figure(f"after_recal_{j}", "png")
