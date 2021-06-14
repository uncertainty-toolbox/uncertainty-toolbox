"""
Examples of code for visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt

import uncertainty_toolbox.viz as uviz
import uncertainty_toolbox.data as udata

import neatplot


# Set plot style
neatplot.set_style()
neatplot.update_rc("figure.dpi", 150)
neatplot.update_rc("text.usetex", False)

# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
(y_pred, y_std, y_true, x) = udata.synthetic_arange_random(20)

# Plot xy
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_xy(y_pred, y_std, y_true, x, leg_loc=4, ax=ax)
neatplot.save_figure('viz_minimal_00', 'png')
plt.show()

# Plot intervals
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_intervals(y_pred, y_std, y_true, ax=ax)
neatplot.save_figure('viz_minimal_01', 'png')
plt.show()

# Plot intervals_ordered
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax)
neatplot.save_figure('viz_minimal_02', 'png')
plt.show()

# Plot calibration
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_calibration(y_pred, y_std, y_true, ax=ax)
neatplot.save_figure('viz_minimal_03', 'png')
plt.show()

# Plot adversarial group calibration
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
uviz.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax)
neatplot.save_figure('viz_minimal_04', 'png')
plt.show()

# Plot sharpness
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_sharpness(y_std, ax=ax)
neatplot.save_figure('viz_minimal_05', 'png')
plt.show()

# Plot residuals vs stds
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
y_resid = y_true - y_pred
uviz.plot_residuals_vs_stds(y_resid, y_std, ax=ax)
neatplot.save_figure('viz_minimal_06', 'png')
plt.show()
