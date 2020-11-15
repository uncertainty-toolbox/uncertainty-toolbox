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
> Confidence calibration is "the problem of predicting probability estimates representative of the true correctness likelihood." [(Guo et al., 2017)](https://arxiv.org/pdf/1706.04599.pdf).

Let <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;\hat{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\hat{Y}" title="\hat{Y}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;\hat{P}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\hat{P}" title="\hat{P}" /></a> be the predicted class and its associated confidence (probability of correctness). We would like the confidence estimates <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;\hat{P}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\hat{P}" title="\hat{P}" /></a> to be calibrated, which intuitively means that we want <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;\hat{P}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\hat{P}" title="\hat{P}" /></a> to represent true probabilities (Guo et al., 2017).

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\mathbb{P}(\hat{Y}=Y&space;\mid&space;\hat{P}=p)=p,&space;\forall&space;p&space;\in[0,1]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;\mathbb{P}(\hat{Y}=Y&space;\mid&space;\hat{P}=p)=p,&space;\forall&space;p&space;\in[0,1]" title="\mathbb{P}(\hat{Y}=Y \mid \hat{P}=p)=p, \forall p \in[0,1]" /></a>
</p>

Suppose a classification model is given <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;N" title="N" /></a> input examples, and made predictions <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;\hat{y}_1,&space;...,&space;\hat{y}_N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\hat{y}_1,&space;...,&space;\hat{y}_N" title="\hat{y}_1, ..., \hat{y}_N" /></a> , each with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;\hat{p}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{300}&space;\hat{p}" title="\hat{p}" /></a> = `0.35`. We would expect `35%` of the predictions would be correct.


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
