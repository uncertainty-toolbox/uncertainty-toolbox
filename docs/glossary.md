# Glossary

A small glossary of key terms for predictive uncertainty quantification.

## Uncertainty
[Wikipedia](https://en.wikipedia.org/wiki/Uncertainty) describes uncertainty as

> "The lack of certainty, a state of limited knowledge where it is impossible to exactly describe the existing state,
> a future outcome, or more than one possible outcome."

In the context of machine learning, uncertainty refers to the lack of confidence in one's estimates.
It is common to separate uncertainty into two categories: [aleatoric uncertainty](#aleatoric-uncertainty)
(inherent uncertainty of the system) and [epistemic uncertainty](#epistemic-uncertainty) (uncertainty about the choice
of model). These two classes of uncertainty are described more below.

Uncertainty is an important consideration in many machine learning applications, and in particular, it may be
crucial to quantify the amount of uncertainty for any given prediction (i.e. the [predictive uncertainty](#predictive-uncertainty)).
When high stakes decisions are being made based on predictions of machine learning models, e.g. in health care, it is 
vital to know how much confidence to have in the prediction. Alternatively, for some sequential decision making tasks
high uncertainty may correspond with potentially valuable decisions that should be tested.

## Predictive Uncertainty
Predictive uncertainty refers to the uncertainty in a prediction made about some target
variable of interest.

As [early review on predictive uncertainty](http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf) outlined the concept by motivating the *task of expressing* predictive uncertainty:
> "One particular approach to expressing uncertainty is to treat the unknown quantity of 
> interest as a random variable, and to make predictions in the form 
> of probability distributions, also known as *predictive distributions*"

To summarize, predictive uncertainty exists whenever one has uncertainty in making a
given prediction, and to express this uncertainty, one can make *distributional
predictions*, instead of *point predictions*.

For example, weather forecasters are tasked with making predictions about incoming
weather conditions.  While various models and analysis can aid in making an educated
prediction, it may be impossible to predict the amount of rainfall exactly. Therefore,
instead of predicting that there will be 0.5 inches of rain tomorrow (a point
prediction), the forecaster can predict that the amount of rain tomorrow is
approximately distributed according to a Gaussian distribution, with mean of 0.5 inches
and standard deviation of 0.05 inches.

Assuming a distributional prediction, the predicted probability attributed to a target quantity is 
often referred to as the [confidence](#Confidence).
Various metrics can be used to evaluate predictive uncertainty, based measures such as
[calibration](#Calibration), [sharpness](#Sharpness), and [proper scoring
rules](#Proper-Scoring-Rules), and [accuracy](#Accuracy).


## Confidence
Following the definition of [predictive uncertainty](#Predictive-Uncertainty), 
assuming we make distributional predictions, the confidence of a prediction is the 
probability attributed to that prediction according to the predicted distribution 
[(Guo et al., 2017)](docs/paper_list.md#:~:text=On%20Calibration%20of%20Modern%20Neural%20Networks).

For example, suppose our task is to predict the match outcome of the world's greatest tennis player, [Nick Kyrgios](https://youtu.be/RaqRV9Kpy9A?t=6), 
and we're interested in predicting the binary outcome, win or lose.
We make an accurate prediction with a Bernoulli distribution that attributes 85% probability of loss (and 15% probability of a win).
Then our *confidence* in predicting a loss is 85%.

This notion of confidence can be extended to multiclass prediction, where we output a categorical 
distribution over the target classes. 
The probability attributed to the mode can be considered the confidence in predicting the 
mode class of the categorical distribution.



## Aleatoric Uncertainty
From [Wikipedia:](https://en.wikipedia.org/wiki/Uncertainty_quantification#:~:text=Aleatoric%20and%20epistemic%20uncertainty,-Uncertainty%20is%20sometimes&text=Aleatoric%20uncertainty%20is%20also%20known,we%20run%20the%20same%20experiment.&text=Epistemic%20uncertainty%20is%20also%20known,but%20do%20not%20in%20practice.)

> "Aleatoric uncertainty is also known as statistical uncertainty, and is representative of unknowns that differ each time we run the same experiment. For example, a single arrow shot with a mechanical bow that exactly duplicates each launch (the same acceleration, altitude, direction and final velocity) will not all impact the same point on the target due to random and complicated vibrations of the arrow shaft, the knowledge of which cannot be determined sufficiently to eliminate the resulting scatter of impact points."

In other words, this is the uncertainty that is inherent to the system because of information that cannot be measured (i.e. noise). When this noise is present, aleatoric uncertainty cannot be eliminated even as the number of samples collected tends towards infinity.

Examples of aleatoric uncertainty can be found in physical settings where measurement error may exist (e.g. given the same true temperature, a thermometer may output slightly different values), or where inherent noise exist in the system (e.g. a random roll of a die). Taking more temperature measurements will not reduce the uncertainty stemming from an imprecise thermometer, and likewise, observing more rolls of a die will not help us better guess the outcome of a roll, assuming the die is fair.


## Epistemic Uncertainty
From [Wikipedia:](https://en.wikipedia.org/wiki/Uncertainty_quantification#:~:text=Aleatoric%20and%20epistemic%20uncertainty,-Uncertainty%20is%20sometimes&text=Aleatoric%20uncertainty%20is%20also%20known,we%20run%20the%20same%20experiment.&text=Epistemic%20uncertainty%20is%20also%20known,but%20do%20not%20in%20practice.)

> "Epistemic uncertainty is also known as systematic uncertainty, and is due to things one could in principle know but do not in practice."

Put another way, epistemic uncertainty is the uncertainty that comes from being unsure about one's model choice. For example, if one is doing modelling with a neural network and is given a finite number of samples to train on, the uncertainty of what the weights in the network should be is epistemic uncertainty. However, as the number of samples being trained on tends to infinity, the epistemic uncertainty tends towards zero as the correct model is able to be identified.

## Calibration

> Confidence calibration is "the problem of predicting probability estimates representative of the 
> true correctness likelihood." 
> [(Guo et al., 2017)](docs/paper_list.md#:~:text=On%20Calibration%20of%20Modern%20Neural%20Networks).

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

## Sharpness

> Sharpness refers to the concentration of the predictive distributions and is a
> property of the forecasts only [(Gneiting et al.)](docs/paper_list.md#:~:text=Probabilistic%20forecasts%2C%20calibration%20and%20sharpness)

Sharpness is a measure of how narrow, concentrated, or peaked the predictive distribtion
is.  Sharpness is evaluated *solely* based on the predictive distribution, and neither
the datapoint nor the ground truth distribution are considered when measuring sharpness.
As an example, a Gaussian distributional prediction with mean 1 and variance 0.5 is
sharper than a prediction with mean 1 and variance 3. 

Sharpness is a valuable property because when the predictive distribution has correct
calibration, a sharper distributional prediction is tighter around the observed
datapoints and thus signifies more confidence in its predictions.

## Proper Scoring Rules

Proper scoring rules are a scalar summary measure of the performance of a distributional prediction.
According to [this seminal work (Gneiting and Raftery)](docs/paper_list.md#:~:text=Strictly%20Proper%20Scoring%20Rules,%20Prediction,%20and%20Estimation),
a proper scoring rule is any function (with mild conditions) that assigns a score to a
predictive probability distribution, where the maximum score of the function is attained
when the predictive distribution exactly matches the ground truth distribution (i.e. the
distribution of the data).

Given this definition, there are many different examples of proper scoring rules, and
different scoring rules for different representations of the predictive distribution.

If the predictive distribution is expressed via a density function, a common scoring
rule we can use is the log-likelihood.  The continuous ranked probability score (CRPS)
is another general score for continuous distribution predictions.  For quantile outputs,
the check score is an applicable proper scoring rule.  The check score is also known as
the "pinall loss" and optimized in standard quantile regression.  The interval score is
a proper scoring rule for centered prediction intervals. Uncertainty Toolbox includes
each of these [scoring rules](uncertainty_toolbox/metrics_scoring_rule.py).


## Accuracy
The accuracy of a prediction is how close it is to the true value. Even if a model achieves good performance with some uncertainty metrics,
it may be useless if it is not accurate. For example, consider a weather model that predicts the temperature for the upcoming day.
This model happens to be perfectly calibrated but unfortunately always predicts the same temperature. If the temperature varies greatly from
day to day, this model will be highly inaccurate and useless to those who want to plan their day out depending on the weather.
