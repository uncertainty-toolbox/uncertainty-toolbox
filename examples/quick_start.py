import uncertainty_toolbox as ut

# Load an example dataset of n=100 predictions, uncertainties, and observations
predictions, predictions_std, y, x = ut.data.synthetic_sine_heteroscedastic(100)

# Compute all uncertainty metrics
metrics = ut.metrics.get_all_metrics(predictions, predictions_std, y)
