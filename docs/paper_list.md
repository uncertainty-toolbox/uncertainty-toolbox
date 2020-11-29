# List of Papers Focusing on Metrics for Predictive Uncertainty Quantification and Probabilistic Forecasting

### Seminary works by Tilmann Gneiting and Co
* Probabilistic forecasts, calibration and sharpness
    - By Tilmann Gneiting, Fadoua Balabdaoui, and Adrian E. Raftery
    - 2007
    - [[Link 1]](https://hal.archives-ouvertes.fr/hal-00363242/document), [[Link 2]](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf)
    - Definitions of calibration and sharpness, and corresponding metrics

* Strictly Proper Scoring Rules, Prediction, and Estimation
    - By Tilmann Gneiting, and Adrian E. Raftery
    - 2007
    - [[Link 1]](https://www.tandfonline.com/doi/pdf/10.1198/016214506000001437), [[Link 2]](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
    - Definition of proper scoring rules and various scoring rules

* Using Bayesian Model Averaging to Calibrate Forecast Ensembles
    - By Adrian E. Raftery, Tilmann Gneiting, Fadoua Balabdaoui, and Michael Polakowski
    - 2005
    - [[Link 1]](https://journals.ametsoc.org/doi/pdf/10.1175/MWR2906.1), [[Link 2]](https://journals.ametsoc.org/doi/full/10.1175/MWR2906.1)

* Using Proper Divergence Functions to Evaluate Climate Models
    - By Thordis L. Thorarinsdottir, Tilmann Gneiting, and Nadine Gissibl
    - 2013
    - [[Link 1]](https://arxiv.org/pdf/1301.5927.pdf), [[Link 2]](https://epubs.siam.org/doi/abs/10.1137/130907550) 

* Probabilistic Forecasting
    - By Tilmann Gneiting and Matthias Katzfuss
    - 2014
    - [[Link 1]](https://www.annualreviews.org/doi/pdf/10.1146/annurev-statistics-062713-085831), [[Link 2]](https://www.annualreviews.org/doi/full/10.1146/annurev-statistics-062713-085831)

### In Machine Learning Literature
* Individual Calibration with Randomized Forecasting
    - By Shengjia Zhao, Tengyu Ma and Stefano Ermon
    - 2020
    - [[Link 1]](), [[Link 2]](https://arxiv.org/pdf/2006.10288.pdf)
    - Calibration; Regression 

* Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification
    - By Youngseog Chung, Willie Neiswanger, Ian Char, and Jeff Schneider
    - 2020
    - [[Link 1]](https://arxiv.org/pdf/2011.09588.pdf), [[Link 2]]()
    - Calibration, Sharpness, Proper Scoring Rules; Regression
   
* On Calibration of Modern Neural Networks
    - By Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger
    - 2017
    - [[Link 1]](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf) [[Link 2]](https://arxiv.org/abs/1706.04599)
    - Calibration and Recalibration; Classification

* Accurate Uncertainties for Deep Learning Using Calibrated Regression
    - By Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon
    - 2018
    - [[Link 1]](http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf) [[Link 2]](https://arxiv.org/abs/1807.00263)
    - Calibration and Recalibration; Regression
    
* Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods
    - By Eyke Hüllermeier and Willem Waegeman
    - 2020
    - [[Link 1]](https://arxiv.org/pdf/1910.09457.pdf)
    - Overview of Concepts, Methods, and Metrics in UQ


<!---
#### Probabilistc Neural Networks
* Estimating the Mean and Variance of the Target Probability Distribution
    - By David A. Nix and Andreas S. Weigend
    - 1994
    - [[Link 1]](https://ieeexplore.ieee.org/abstract/document/374138), [[Link 2]]()
    - Seminal work on optimizing a likelihood-based loss with neural networks; Regression 
    
* Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
    - By Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell
    - 2017
    - [[Link 1]](https://arxiv.org/abs/1612.01474), [[Link 2]](https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
    - Classification & Regression

* Individual Calibration with Randomized Forecasting
    - By Shengjia Zhao, Tengyu Ma and Stefano Ermon
    - 2020
    - [[Link 1]](), [[Link 2]](https://arxiv.org/pdf/2006.10288.pdf)
    - Regression 

* Reliable Training and Estimation of Variance Networks
    - By Nicki S. Detlefsen, Martin Jørgensen, and Søren Hauberg
    - 2019
    - [[Link 1]](https://arxiv.org/pdf/1906.03260.pdf), [[Link 2]](https://papers.nips.cc/paper/2019/file/07211688a0869d995947a8fb11b215d6-Paper.pdf)
    - Regression

#### Bayesian Methods
##### Bayesian Neural Networks (BNN)
* Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users
    - By Laurent Valentin Jospin, Wray Buntine, Farid Boussaid, Hamid Laga, and Mohammed Bennamoun
    - 2020
    - [[Link 1]](https://arxiv.org/pdf/2007.06823.pdf), [[Link 2]]()
    - Comprehensive tutorial on various methods for BNN
    
* How Good is the Bayes Posterior in Deep Neural Networks Really?
    - By Florian Wenzel, Kevin Roth, Bastiaan S. Veeling, Jakub Świątkowski, Linh Tran, Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton, and Sebastian Nowozin
    - 2020
    - [[Link 1]](https://proceedings.icml.cc/static/paper_files/icml/2020/3581-Paper.pdf), [[Link 2]](https://arxiv.org/pdf/2002.02405.pdf)
    - Empirical study on the performance of posterior of BNN

* Stochastic Gradient Hamiltonian Monte Carlo
    - By Tianqi Chen, Emily B. Fox, and Carlos Guestrin
    - 2014
    - [[Link 1]](http://proceedings.mlr.press/v32/cheni14.pdf), [[Link 2]](https://arxiv.org/pdf/1402.4102.pdf)
    - Markov Chain Monte Carlo (MCMC) based; Classification & Regression

* Weight Uncertainty in Neural Networks
    - By Charles Blundell, Julien Cornebise, Koray Kavukcuoglu and Daan Wierstra
    - 2015
    - [[Link 1]](http://proceedings.mlr.press/v37/blundell15.pdf), [[Link 2]](https://arxiv.org/pdf/1505.05424.pdf)
    - Variational Inference (VI) based; Classification & Regression

##### Bayesian Modeling
* Predictive Uncertainty Estimation via Prior Networks
    - By Andrey Malinin and Mark Gales
    - 2018
    - [[Link 1]](https://proceedings.neurips.cc/paper/2018/file/3ea2db50e62ceefceaf70a9d9a56a6f4-Paper.pdf), [[Link 2]](https://arxiv.org/pdf/1802.10501.pdf)
    - Classification

##### Bayesian Inference
* A Simple Baseline for Bayesian Uncertainty in Deep Learning
    - By Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov and Andrew Gordon Wilson
    - 2019
    - [[Link 1]](https://papers.nips.cc/paper/2019/file/118921efba23fc329e6560b27861f0c2-Paper.pdf), [[Link 2]](https://arxiv.org/pdf/1902.02476.pdf)
    - Classification & Regression

* Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
    - By Yarin Gal and Zoubin Ghahramani
    - 2016
    - [[Link 1]](http://proceedings.mlr.press/v48/gal16.pdf), [[Link 2]](https://arxiv.org/pdf/1506.02142.pdf)
    - Classification & Regression

#### Quantile Based Methods
* Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification
    - By Youngseog Chung, Willie Neiswanger, Ian Char, and Jeff Schneider
    - 2020
    - [[Link 1]](https://arxiv.org/pdf/2011.09588.pdf), [[Link 2]]()
    - Regression

* Single-Model Uncertainties for Deep Learning
    - By Natasa Tagasovska and David Lopez-Paz
    - 2019
    - [[Link 1]](https://papers.nips.cc/paper/2019/file/73c03186765e199c116224b68adc5fa0-Paper.pdf), [[Link 2]](https://arxiv.org/pdf/1811.00908.pdf)
    - Classification & Regression
    
* High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach
    - By Tim Pearce, Mohamed Zaki, Alexandra Brintrup and Andy Neely
    - 2018
    - [[Link 1]](http://proceedings.mlr.press/v80/pearce18a/pearce18a.pdf), [[Link 2]](https://arxiv.org/pdf/1802.07167.pdf)
    - Regression
   
#### Holistic Review of Methods
* Methods for Comparing Uncertainty Quantifications for Material Property Predictions
    - By Kevin Tran, Willie Neiswanger, Junwoong Yoon, Qingyang Zhang, Eric Xing, and Zachary W Ulissi
    - 2020
    - [[Link 1]](https://iopscience.iop.org/article/10.1088/2632-2153/ab7e1a/pdf), [[Link 2]]()
    - Empirical Comparison of UQ Methods; Regression
   
* Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift
    - By Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek
    - 2019
    - [[Link 1]](https://proceedings.neurips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf), [[Link 2]](https://arxiv.org/pdf/1906.02530.pdf)
    - Empirical Comparison of UQ Methods; Classification

* Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods
    - By Eyke Hüllermeier and Willem Waegeman
    - 2020
    - [[Link 1]](https://arxiv.org/pdf/1910.09457.pdf)
    - Overview of Concepts and Methods in UQ
   
#### Recalibration 
* On Calibration of Modern Neural Networks
    - By Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger
    - 2017
    - [[Link 1]](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf) [[Link 2]](https://arxiv.org/abs/1706.04599)
    - Classification

* Accurate Uncertainties for Deep Learning Using Calibrated Regression
    - By Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon
    - 2018
    - [[Link 1]](http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf) [[Link 2]](https://arxiv.org/abs/1807.00263)
    - Regression


* 
    - By 
    - 
    - [[Link 1]](), [[Link 2]]()
    - 


As a primer to get familiar with the concepts and objectives of uncertainty quantification, we recommend starting off with the following reading list:

* Proper scoring rules
    - Gneiting's paper defining proper scoring rules
* Calibration and sharpness metrics
    - Gneiting's 
    - Preliminaries of Beyond Pinball Loss
    - Preliminaries of Individual Calibration
*  Overview of Current Methods
    - Hüllermeier's Introductory Paper
    
--->