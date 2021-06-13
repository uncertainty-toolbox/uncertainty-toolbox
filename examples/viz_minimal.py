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
(y_pred, y_std, y_true, _) = udata.synthetic_arange_random()

# Plot intervals
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_intervals(y_pred, y_std, y_true, ax=ax)
neatplot.save_figure('viz_minimal_01', 'svg')
plt.show()

# Plot calibration
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_calibration(y_pred, y_std, y_true, ax=ax)
neatplot.save_figure('viz_minimal_02', 'svg')
plt.show()

# Plot intervals_ordered
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
uviz.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax)
neatplot.save_figure('viz_minimal_03', 'svg')
plt.show()
