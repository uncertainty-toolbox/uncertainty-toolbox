"""
Examples of code for visualizations.
"""

import numpy as np

import uncertainty_toolbox.viz as uviz
import uncertainty_toolbox.data as udata


# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
(y_pred, y_std, y_true) = udata.synthetic_arange_random()

# Print details about the synthetic results
print('* y_true: {}'.format(y_true))
print('* y_pred: {}'.format(y_pred))
print('* |y_true - y_pred|: {}'.format(np.abs(y_true - y_pred)))
print('* y_std: {}'.format(y_std))

# Plot
uviz.plot_parity(y_pred, y_true, show=True)
uviz.plot_intervals(y_pred, y_std, y_true, show=True)
uviz.plot_calibration(y_pred, y_std, y_true, show=True)
