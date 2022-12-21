"""Uncertainty Toolbox: a python toolbox for predictive uncertainty quantification, calibration, metrics, and visualization."""

__version__ = "0.1.1rc1"

from .data import (
    synthetic_arange_random,
    synthetic_sine_heteroscedastic,
)

from .metrics import (
    get_all_metrics,
    get_all_accuracy_metrics,
    get_all_average_calibration,
    get_all_adversarial_group_calibration,
    get_all_sharpness_metrics,
    get_all_scoring_rule_metrics,
)

from .metrics_accuracy import prediction_error_metrics

from .metrics_calibration import (
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    miscalibration_area,
    adversarial_group_calibration,
    sharpness,
    get_proportion_lists_vectorized,
    get_proportion_lists,
    get_proportion_in_interval,
    get_prediction_interval,
)

from .metrics_scoring_rule import (
    nll_gaussian,
    crps_gaussian,
    check_score,
    interval_score,
)

from .recalibration import (
    iso_recal,
    optimize_recalibration_ratio,
)

from .viz import (
    plot_xy,
    plot_intervals,
    plot_intervals_ordered,
    plot_calibration,
    plot_adversarial_group_calibration,
    plot_sharpness,
    plot_residuals_vs_stds,
)
