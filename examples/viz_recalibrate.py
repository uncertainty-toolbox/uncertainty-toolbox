"""
Examples of code for recalibration.
"""

import numpy as np
import matplotlib.pyplot as plt

import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
from uncertainty_toolbox.metrics_calibration import get_proportion_lists_vectorized
import uncertainty_toolbox.viz as uviz
from uncertainty_toolbox.recalibration import iso_recal

import neatplot


# Set plot style
neatplot.set_style()
neatplot.update_rc("text.usetex", True)     # Set to True for system latex
neatplot.update_rc("font.size", 14)         # Set font size
neatplot.update_rc("xtick.labelsize", 14)   # Set font size for xaxis tick labels
neatplot.update_rc("ytick.labelsize", 14)   # Set font size for yaxis tick labels

# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
n_obs = 650
f, std, y, x = udata.synthetic_sine_heteroscedastic(n_obs)

# Save figure (set to True to save)
savefig = True

# List of predictive means and standard deviations
pred_mean_list = [f]

pred_std_list = [
    std * 0.5,  # overconfident
    std * 2.0,  # underconfident
]

# Loop through, make plots, and compute metrics
for i, pred_mean in enumerate(pred_mean_list):
    for j, pred_std in enumerate(pred_std_list):

        # Before recalibration
        exp_props, obs_props = get_proportion_lists_vectorized(pred_mean, pred_std, y)
        mace = umetrics.mean_absolute_calibration_error(
            pred_mean, pred_std, y, recal_model=None
        )
        rmsce = umetrics.root_mean_squared_calibration_error(
            pred_mean, pred_std, y, recal_model=None
        )
        ma = umetrics.miscalibration_area(pred_mean, pred_std, y, recal_model=None)
        print("Before Recalibration:  ", end='')
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        uviz.plot_calibration(
            pred_mean,
            pred_std,
            y,
            exp_props=exp_props,
            obs_props=obs_props,
            ax=ax,
        )
        neatplot.save_figure(f'before_recal_{j}', 'svg')

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
        print("After Recalibration:  ", end='')
        print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        uviz.plot_calibration(
            pred_mean,
            pred_std,
            y,
            exp_props=recal_exp_props,
            obs_props=recal_obs_props,
            ax=ax,
        )
        neatplot.save_figure(f'after_recal_{j}', 'svg')
