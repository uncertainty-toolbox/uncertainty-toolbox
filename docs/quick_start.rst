###########
Quick Start
###########

This is a quick start tutorial for Uncertainty Toolbox. For a longer tutorial, `see here
<https://uncertainty-toolbox.github.io/tutorial>`_.

***************
Minimal example
***************

This example computes a set of metrics for a vector of predicted values (`predictions`)
and associated uncertainties (`predictions_std`, a vector of standard deviations), taken
with respect to a corresponding set of observed values `y`.

.. code-block:: python

  import uncertainty_toolbox as uct

  # Load an example dataset of 100 predictions, uncertainties, and observations
  predictions, predictions_std, y, x = uct.data.synthetic_sine_heteroscedastic(100)

  # Compute all uncertainty metrics
  metrics = uct.metrics.get_all_metrics(predictions, predictions_std, y)
