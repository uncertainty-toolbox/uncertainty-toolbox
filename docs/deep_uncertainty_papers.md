# Deep (and Other) Predictive Uncertainty Quantification Papers


## Ensembles 

<!---
* Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
  - By Yarin Gal, Zoubin Ghahramani
  - [Arxiv paper](https://arxiv.org/abs/1506.02142)
---> 

* Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
  - By Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell
  - [Arxiv paper](https://arxiv.org/abs/1612.01474)

* **Related**: Strictly Proper Scoring Rules, Prediction, and Estimation
  - By Tilmann Gneiting and Adrian E. Raftery
  - [Paper pdf](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), see also [wiki page](https://en.wikipedia.org/wiki/Scoring_rule)

* Uncertainty in Neural Networks: Bayesian Ensembling
  - By Tim Pearce, Mohamed Zaki, Alexandra Brintrup, Nicolas Anastassacos, Andy Neely
  - [Arxiv paper](https://arxiv.org/abs/1810.05546)
  
* Ensemble methods in Machine Learning
  - By Thomas G. Dietterich
  - [Document link](http://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf)

* Representation of Uncertainty in Deep Neural Networks through Sampling
  - By Patrick McClure, Nikolaus Kriegeskorte
  - [Paper pdf](https://openreview.net/references/pdf?id=HJ1JBJ5gl)

* Ensemble Sampling
  - By Xiuyuan Lu, Benjamin Van Roy
  - [Arxiv paper](https://arxiv.org/pdf/1705.07347.pdf)

* Maximizing Overall Diversity for Improved Uncertainty Estimates in Deep Ensembles
  - By Siddhartha Jain, Ge Liu, Jonas Mueller, David Gifford
  - [Arxiv paper](https://arxiv.org/pdf/1906.07380.pdf)

## Calibration and Recalibration

* Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification
    - By Youngseog Chung, Willie Neiswanger, Ian Char, Jeff Schneider
    - [Arxiv paper](https://arxiv.org/pdf/2011.09588.pdf)

* Individual Calibration with Randomized Forecasting
  - By Shengjia Zhao, Tengyu Ma, Stefano Ermon
  - [Arxiv paper](https://arxiv.org/pdf/2006.10288.pdf) 

* On Calibration of Modern Neural Networks
  - By Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
  - [Arxiv paper](https://arxiv.org/abs/1706.04599)

* Accurate Uncertainties for Deep Learning Using Calibrated Regression
  - By Volodymyr Kuleshov, Nathan Fenner, Stefano Ermon
  - [Arxiv paper](https://arxiv.org/abs/1807.00263)

* Single-Model Uncertainties for Deep Learning
    - By Natasa Tagasovska, David Lopez-Paz
    - [Arxiv paper](https://arxiv.org/pdf/1811.00908.pdf)
    
* High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach
    - By Tim Pearce, Mohamed Zaki, Alexandra Brintrup and Andy Neely
    - [Arxiv paper](https://arxiv.org/pdf/1802.07167.pdf)


## Bayesian Methods
### Bayesian Neural Networks (BNN)
* Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users
  - By Laurent Valentin Jospin, Wray Buntine, Farid Boussaid, Hamid Laga, Mohammed Bennamoun
  - Comprehensive tutorial on various methods for BNN
  - [Arxiv paper](https://arxiv.org/pdf/2007.06823.pdf)

* Bayesian Learning for Neural Networks 
  - By Radford M. Neal
  - Seminal work on Markov Chain Monte Carlo (MCMC) based learning for neural networks
  - [Paper pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.446.9306&rep=rep1&type=pdf)

* Stochastic Gradient Hamiltonian Monte Carlo
  - By Tianqi Chen, Emily B. Fox, Carlos Guestrin
  - MCMC based
  - [Arxiv paper](https://arxiv.org/pdf/1402.4102.pdf)

* Bayesian Learning via Stochastic Gradient Langevin Dynamics
  - By Max Welling, Yee Whye Teh
  - MCMC based
  - [Paper pdf](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)

* Weight Uncertainty in Neural Networks
  - By Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra
  - Variational Inference (VI) based
  - [Arxiv paper](https://arxiv.org/pdf/1505.05424.pdf)

* Deterministic Variational Inference for Robust Bayesian Neural Networks
  - By Anqi Wu, Sebastian Nowozin, Edward Meeds, Richard E. Turner, José Miguel Hernández-Lobato, Alexander L. Gaunt
  - VI based
  - [Arxiv paper](https://arxiv.org/pdf/1810.03958.pdf)

* Noisy Natural Gradient as Variational Inference
  - Guodong Zhang, Shengyang Sun, David Duvenaud, Roger Grosse
  - VI based
  - [Arxiv paper](https://arxiv.org/pdf/1712.02390.pdf)

* Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam
  - Mohammad Emtiyaz Khan, Didrik Nielsen, Voot Tangkaratt, Wu Lin, Yarin Gal, Akash Srivastava
  - VI based
  - [Arxiv paper](https://arxiv.org/pdf/1806.04854.pdf)

* Noise Contrastive Priors for Functional Uncertainty
  - By Danijar Hafner, Dustin Tran, Timothy Lillicrap, Alex Irpan, James Davidson
  - [Arxiv paper](https://arxiv.org/abs/1807.09289)
  
* Bayesian Layers: A Module for Neural Network Uncertainty
  - By Dustin Tran, Michael W. Dusenberry, Mark van der Wilk, Danijar Hafner
  - [Arxiv paper](https://arxiv.org/abs/1812.03973)

<!---  
### Bayesian Modeling
* Predictive Uncertainty Estimation via Prior Networks
  - By Andrey Malinin, Mark Gales
  - [Arxiv paper](https://arxiv.org/pdf/1802.10501.pdf)
--->

### Bayesian Inference
* A Simple Baseline for Bayesian Uncertainty in Deep Learning
  - By Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson
  - [Arxiv paper](https://arxiv.org/pdf/1902.02476.pdf)

* Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
  - By Yarin Gal, Zoubin Ghahramani
  - [Arxiv paper](https://arxiv.org/pdf/1506.02142.pdf)

## Deep GPs, Deep Kernels, and Neural Processes

* Deep Kernel Learning
  - By Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, Eric P. Xing
  - [Arxiv paper](https://arxiv.org/abs/1511.02222)

* Neural Processes
  - By Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh
  - [Arxiv paper](https://arxiv.org/abs/1807.01622)

* On the Connection between Neural Processes and Gaussian Processes with Deep Kernels
  - By Tim GJ Rudner, Vincent Fortuin, Yee Whye Teh, Yarin Gal
  - [NeurIPS Workshop paper](http://bayesiandeeplearning.org/2018/papers/128.pdf)


## Meta-Network Strategies
### (Training a neural network or other model to directly predict confidence/uncertainty.)

* Note: see what is used for each element of ensemble in "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles".

* Predictive Uncertainty Estimation via Prior Networks
  - By Andrey Malinin, Mark Gales
  - [NeurIPS site](http://papers.nips.cc/paper/7936-predictive-uncertainty-estimation-via-prior-networks)

* Learning Confidence for Out-of-Distribution Detection in Neural Networks
  - By Terrance DeVries, Graham W. Taylor
  - [Arxiv paper](https://arxiv.org/abs/1802.04865)

* Learning for Single-Shot Confidence Calibration in Deep Neural Networks through Stochastic Inferences
  - By Seonguk Seo, Paul Hongsuck Seo, Bohyung Han
  - [Arix paper](https://arxiv.org/abs/1809.10877)

* Detecting Adversarial Examples and Other Misclassifications in Neural Networks by Introspection
  - By Jonathan Aigrain, Marcin Detyniecki
  - [Arxiv paper](https://arxiv.org/abs/1905.09186)

* Towards Better Confidence Estimation for Neural Models
  - By Vishal Thanvantri Vasudevan, Abhinav Sethy, Alireza Roshan Ghias
  - [Paper pdf](https://d39w7f4ix9f5s9.cloudfront.net/ae/5d/24a9ac264d34bb73f313f2713f89/scipub-133.pdf)

* Density estimation in representation space to predict model uncertainty
  - By Tiago Ramalho, Miguel Miranda
  - [Arxiv paper](https://arxiv.org/abs/1908.07235)

* Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples
  - By Kimin Lee, Honglak Lee, Kibok Lee, Jinwoo Shin
  - [Arxiv paper](https://arxiv.org/abs/1711.09325)

* Uncertainty Estimation Using a Single Deep Deterministic Neural Network
  - By Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal
  - [Arxiv paper](https://arxiv.org/pdf/2003.02037.pdf)


## Holistic Review of UQ Methods
* Methods for Comparing Uncertainty Quantifications for Material Property Predictions
  - By Kevin Tran, Willie Neiswanger, Junwoong Yoon, Qingyang Zhang, Eric Xing, Zachary W Ulissi
  - Empirical comparison of UQ methods for regression
  - [Arxiv paper](https://arxiv.org/pdf/1912.10066.pdf),
   
* Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift
  - By Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek
  - Empirical comparison of UQ methods under dataset shift, for classification
  - [Arxiv paper](https://arxiv.org/pdf/1906.02530.pdf)
  
* Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods
  - By Eyke Hüllermeier and Willem Waegeman
  - Overview of concepts and methods in UQ
  - [Arxiv paper](https://arxiv.org/pdf/1910.09457.pdf)
  

<!---
## Other Papers
### (Some of these should potentially be sorted into other categories.)


* Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift
  - By Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek
  - [Arxiv paper](https://arxiv.org/abs/1906.02530)
--->


## Downstream applications
### Computer Vision
* Deep Bayesian Active Learning with Image Data
  - By Yarin Gal, Riashat Islam, Zoubin Ghahramani
  - [Arxiv paper](https://arxiv.org/abs/1703.02910)

* What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
  - By Alex Kendall, Yarin Gal
  - [Arxiv paper](https://arxiv.org/abs/1703.04977)

### Reinforcement Learning
* Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
  - Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine
  - [Arxiv paper](https://arxiv.org/pdf/1805.12114.pdf)

* MOPO: Model-based Offline Policy Optimization
  - By Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, Tengyu Ma
  - [Arxiv paper](https://arxiv.org/pdf/2005.13239.pdf)
  
* When to Trust Your Model: Model-Based Policy Optimization
  - By Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine
  - [Arxiv paper](https://arxiv.org/pdf/1906.08253.pdf)
  
* Calibrated Model-Based Deep Reinforcement Learning
  - By Ali Malik, Volodymyr Kuleshov, Jiaming Song, Danny Nemer, Harlan Seymour, Stefano Ermon
  - [Arxiv paper](https://arxiv.org/abs/1906.08312)

### Language
* Incorporating Uncertainty into Deep Learning for Spoken Language Assessment
  - By Andrey Malinin, Anton Ragni, Kate M. Knill, Mark J. F. Gales
  - [Paper pdf](https://www.aclweb.org/anthology/P17-2008.pdf)


<!---
## Deep (classic) approximate Bayesian inference
* TODO: recent work on SG-MCMC or VI applied successfully to large deep models.
--->