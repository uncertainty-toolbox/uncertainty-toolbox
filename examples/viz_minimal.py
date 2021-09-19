"""
Examples of code for visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct


# Set plot style
uct.viz.set_style()
uct.viz.update_rc("figure.dpi", 150)
uct.viz.update_rc("text.usetex", False)

# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
(y_pred, y_std, y_true, x) = uct.synthetic_arange_random(20)

# Plot xy
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uct.plot_xy(y_pred, y_std, y_true, x, leg_loc=4, ax=ax)
uct.viz.save_figure("viz_minimal_00", "png")
plt.show()

# Plot intervals
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uct.plot_intervals(y_pred, y_std, y_true, ax=ax)
uct.viz.save_figure("viz_minimal_01", "png")
plt.show()

# Plot intervals_ordered
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax)
uct.viz.save_figure("viz_minimal_02", "png")
plt.show()

# Plot calibration
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uct.plot_calibration(y_pred, y_std, y_true, ax=ax)
uct.viz.save_figure("viz_minimal_03", "png")
plt.show()

# Plot adversarial group calibration
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax)
uct.viz.save_figure("viz_minimal_04", "png")
plt.show()

# Plot sharpness
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uct.plot_sharpness(y_std, ax=ax)
uct.viz.save_figure("viz_minimal_05", "png")
plt.show()

# Plot residuals vs stds
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uct.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax)
uct.viz.save_figure("viz_minimal_06", "png")
plt.show()
