
# Steerable Conditional Neural Processes

In this repository, you can find an implementation of **Steerable Conditional Neural Processes**, a work published at ICML 2021 that you find [here](http://proceedings.mlr.press/v139/holderrieth21a/holderrieth21a.pdf).

Steerable Conditional Neural Processes (SteerCNPs) are an extension of [Conditional Neural Processes](https://arxiv.org/abs/1807.01613). The model consists of an encoder which is based on the work of [Gordon et al (2020)](https://arxiv.org/abs/1910.13556) and a decoder which is an equivariant neural network of the form outlined in the work of [Weiler et al (2019)](https://arxiv.org/abs/1911.08251).

The image below depicts example predictions of the SteerCNP in the case of vector fields. Inputs are the red arrows and the model extracts the whole vector field.

![GP_Predictions](https://github.com/PeterHolderrieth/Steerable_CNPs/blob/master/plots/gp/Example_predictions_SteerCNP.png?raw=true)

## Repository

This library provides an implementation of SteerCNPs and the code for two experiments.

We tested our model on two data sets: a Gaussian process regression task and real-world weather data.
Below, the model predicts the wind in a cyclic region of 500km radius around Memphis in the South of the US.
It gets measurements of wind, temperature and pressure from places marked in red.

![ERA5Predictions](https://github.com/PeterHolderrieth/Steerable_CNPs/blob/master/plots/era5/ERA5_predictions.png?raw=true)

## Installation

To use this repository, one can simply clone it and  install requirements specified in `requirements.txt`.
Please note this installs additional packages used for visualization. In particular, we use:

- [PyTorch](https://https://pytorch.org/) as a library for automatic differentation.
-  the library [E(2)-Steerable CNNs](https://github.com/QUVA-Lab/e2cnn) for any group-related tasks and an implementation of Steerable CNNs.
- the [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) data set giving grided global weather data. We share a full download and pre-processing script to 
extract the data set we use.

## Structure of the repository
The core implementation of SteerCNPs are all files in the root. The folder "tasks" gives the two main tasks (data sets+ data loading scripts) which we have given our model: GP vector field data and
real-world weather data. The folder "experiments" gives the main execution file per task. 
The folder CNP gives an implementation of [Conditional Neural Processes](https://arxiv.org/abs/1807.01613)
to compare our results.

