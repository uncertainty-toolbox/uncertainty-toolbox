# Glossary

A small glossary of key terms for predictive uncertainty quantification.

## Accuracy
**TODO**: define.

## Aleatoric Uncertainty
From [Wikipedia:](https://en.wikipedia.org/wiki/Uncertainty_quantification#:~:text=Aleatoric%20and%20epistemic%20uncertainty,-Uncertainty%20is%20sometimes&text=Aleatoric%20uncertainty%20is%20also%20known,we%20run%20the%20same%20experiment.&text=Epistemic%20uncertainty%20is%20also%20known,but%20do%20not%20in%20practice.)

> "Aleatoric uncertainty is also known as statistical uncertainty, and is representative of unknowns that differ each time we run the same experiment. For example, a single arrow shot with a mechanical bow that exactly duplicates each launch (the same acceleration, altitude, direction and final velocity) will not all impact the same point on the target due to random and complicated vibrations of the arrow shaft, the knowledge of which cannot be determined sufficiently to eliminate the resulting scatter of impact points."

In other words, this is the noise that is inherent to the system because of information that cannot be measured. When this noise is present, aleatoric uncertainty cannot be eliminated even as the number of samples collected tends towards infinity. 

## Calibration
**TODO**: define.

## Confidence
**TODO**: define.

## Epistemic Uncertainty
From [Wikipedia:](https://en.wikipedia.org/wiki/Uncertainty_quantification#:~:text=Aleatoric%20and%20epistemic%20uncertainty,-Uncertainty%20is%20sometimes&text=Aleatoric%20uncertainty%20is%20also%20known,we%20run%20the%20same%20experiment.&text=Epistemic%20uncertainty%20is%20also%20known,but%20do%20not%20in%20practice.)

> "Epistemic uncertainty is also known as systematic uncertainty, and is due to things one could in principle know but do not in practice."

Put another way, epistemic uncertainty is the uncertainty that comes from being unsure about one's model choice. For example, if one is doing modelling with a neural network and is given a finite number of samples to train on, the uncertainty of what the weights in the network should be is epistemic uncertainty. However, as the number of samples being trained on tends to infinity, the epistemic uncertainty tends towards zero as the correct model is able to be identified.

## Predictive Uncertainty
**TODO**: define.

## Sharpness
**TODO**: define.

## Uncertainty
**TODO**: define.
