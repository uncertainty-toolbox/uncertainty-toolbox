"""
Code for the Uncertainty Toolbox
"""

from .data import (
    synthetic_arange_random,
    synthetic_sine_heteroscedastic,
    curvy_cosine,
)

from .metrics import (
    get_all_metrics,
)

from .metrics_accuracy import prediction_error_metrics

from .metrics_calibration import (
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    miscalibration_area,
    adversarial_group_calibration,
    sharpness,
)

from .metrics_scoring_rule import (
    nll_gaussian,
    crps_gaussian,
    check_score,
    interval_score,
)
