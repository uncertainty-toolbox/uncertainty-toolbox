# Uncertainty Toolbox

<br/>

**Uncertainty Toolbox**
> A python toolbox for predictive uncertainty quantification, calibration, metrics, and
> visualization.\
> And a collection of [relevant papers and references](docs/paper_list.md).

<!--**A python toolbox for predictive uncertainty quantification, calibration, metrics, and visualization.**-->

&nbsp;\
Many machine learning methods return predictions along with uncertainties of some form,
such as distributions or confidence intervals. This begs the questions: How do we
determine which predictive uncertanties are best? What does it mean to produce a _best_
or _ideal_ uncertainty?

Uncertainty Toolbox provides standard metrics to quantify the quality of predictive
uncertainties, describes the intuition behind these metrics, produces visualizations of
these metrics/uncertainties, and implements simple post-processing procedures to improve
these uncertainties (i.e.  "re-calibration").  This toolbox is for both classification
and regression tasks. The UCT also aims to provide a reference list of relevant papers
in this area.


## Installation

Tuun requires Python 3.6+. To install, clone and `cd` into this repo, and run:
```
$ pip install -r requirements/requirements.txt
```


## Quick Start
[**TODO**: show some minimal examples of usage here.]


## Full Contents
[**TODO**: somewhere need to give full contents of the toolbox. Should I include it
here, or make a full readthedocs page? I also have a plan to link to various
aspects of the contents via some images near the top of this page.]


## TODOs
* contents: metrics, visualizations, list of papers (+ descriptions?), example code
* unit tests
* documentation (readthedocs?) and examples of code/usage
* pip and pypi
* logo, etc
