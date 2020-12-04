# Glossary

A small glossary of key terms for predictive uncertainty quantification.

## Accuracy
**TODO**: define.

## Aleatoric Uncertainty
From [Wikipedia:](https://en.wikipedia.org/wiki/Uncertainty_quantification#:~:text=Aleatoric%20and%20epistemic%20uncertainty,-Uncertainty%20is%20sometimes&text=Aleatoric%20uncertainty%20is%20also%20known,we%20run%20the%20same%20experiment.&text=Epistemic%20uncertainty%20is%20also%20known,but%20do%20not%20in%20practice.)

> "Aleatoric uncertainty is also known as statistical uncertainty, and is representative of unknowns that differ each time we run the same experiment. For example, a single arrow shot with a mechanical bow that exactly duplicates each launch (the same acceleration, altitude, direction and final velocity) will not all impact the same point on the target due to random and complicated vibrations of the arrow shaft, the knowledge of which cannot be determined sufficiently to eliminate the resulting scatter of impact points."

In other words, this is the uncertainty that is inherent to the system because of information that cannot be measured (i.e. noise). When this noise is present, aleatoric uncertainty cannot be eliminated even as the number of samples collected tends towards infinity.

Examples of aleatoric uncertainty can be found in physical settings where measurement error may exist (e.g. given the same true temperature, a thermometer may output slightly different values), or where inherent noise exist in the system (e.g. a random roll of a die). Taking more temperature measurements will not reduce the uncertainty stemming from an imprecise thermometer, and likewise, observing more rolls of a die will not help us better guess the outcome of a roll, assuming the die is fair.  


## Calibration

> Confidence calibration is "the problem of predicting probability estimates representative of the true correctness likelihood." [(Guo et al., 2017)](https://arxiv.org/pdf/1706.04599.pdf).

Intuitively, calibration refers to the degree to which a predicted uncertainty matches
the true underlying uncertainty in the data.  For example, suppose you make a series of
10 predictions about the winning odds of a horse race, and for each race you predict
that a certain horse will win with 70% chance. If you called the correct winning horse
in roughly 7 of the 10 races (70% of the races), then your uncertainty predictions are
said to be calibrated.  The above example is an example of calibrated classification
with binary outcomes (win or lose).

We can further consider calibration in the regression setting.  If you make predictions
about the amount of rainfall each day for a whole month, and for each day, you make a
predictive statement, _"the amount of rainfall today will not be more than x inches,
with 50% chance"_, and if your prediction was correct for roughly 15 out of 30 days
(50% of the days), then your predictions are said to be calibrated.


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
