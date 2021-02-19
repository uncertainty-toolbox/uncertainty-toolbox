"""
Visualization for Uncertainty Toolbox logo.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
import uncertainty_toolbox.viz as uviz

import neatplot

neatplot.set_style()
neatplot.update_rc('axes.grid', False)
neatplot.update_rc('figure.dpi', 150)


# Set random seed
np.random.seed(11)

# Generate synthetic predictive uncertainty results
n_obs = 650
f, std, y, x = udata.synthetic_sine_heteroscedastic(n_obs)

ylims = [-3, 3]
xlims = [1.3, 10.75]
n_subset = 50

# Make xy plot
uviz.plot_xy(f, std, y, x, n_subset=300, ylims=ylims, xlims=xlims)

# Set size and save
fig = plt.gcf()
fig.set_size_inches(7.0, 3.0)

neatplot.save_figure('xy', ['pdf', 'svg'])
plt.show()
