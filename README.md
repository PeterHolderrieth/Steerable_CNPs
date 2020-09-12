
# Equivariant Conditional Neural Processes

In this repository, you can find an implementation of **Equivariant Conditional Neural Processes**,
a model which I developped as part of my master's dissertation. 
It it a meta-learning method which exploits the geometry of data.

![GP_Predictions](https://github.com/PeterHolderrieth/EquivariantCNPs/blob/master/Evaluation/GP_div_free/Example_predictions_EquivCNP.png?raw=true)

## Model 

Equivariant Conditional Neural Processes are an extension of [Conditional Neural Processes](https://arxiv.org/abs/1807.01613). The image above depicts example
predictions of the EquivCNP in the case of vector fields. Inputs are the red arrows and the model extracts the whole
vector field.

The model consists of an encoder called **EquivDeepSet** and a decoder which is an equivariant neural network 
of the form outlined in the work of [Weiler et al](https://arxiv.org/abs/1911.08251).

## Experiments

We tested our model on two data sets: a Gaussian process regression task and real-world weather data.
Below, the model predicts the wind in a cyclic region of 500km radius around Memphis in the South of the US.
It gets measurements of wind, temperature and pressure from places marked in red.

![ERA5Predictions](https://github.com/PeterHolderrieth/EquivariantCNPs/blob/master/Evaluation/ERA5/ERA5_Predictions_4.png?raw=true)

## Links
- We used [PyTorch](https://https://pytorch.org/) as a library for automatic differentation.
- We made use of the library [E(2)-Steerable CNNs](https://github.com/QUVA-Lab/e2cnn)
for any group-related tasks.
- We used the [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) data set giving grided global weather data. 

## Structure of the repository
The core implementation of EquivCNPs are all files in the root. The folder "Tasks" gives the two main tasks which we have given our model: GP vector field data and
real-world weather data. The folder "Experiments" gives the main execution file per task and "Evaluation"
gives all evaluation scripts. The folder CNP gives an implementation of [Conditional Neural Processes](https://arxiv.org/abs/1807.01613)
to compare our results.

## Acknowledgement 
This project was part my master's dissertation supervised by Professor
Yee Whye Teh (University of Oxford, Google DeepMind) and co-supervised by Michael Hutchinson
(University of Oxford).

## Contact

If you have any questions, feel free to contact me (peter.holderrieth@new.ox.ac.uk).
