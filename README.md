
# Equivariant Conditional Neural Processes

In this repository, you can find an implementation of **Equivariant Conditional Neural Processes**,
a model which I developped as part of my master's dissertation called. 
It it a meta-learning method which exploits the geometry of data.

![alt text](https://github.com/PeterHolderrieth/EquivariantCNPs/blob/master/Evaluation/GP_div_free/Example_predictions_EquivCNP.png?raw=true)

## Model 

Equivariant Conditional Neural Processes are an extension of Conditional Neural Processes. The image above depicts example
predictions of the EquivCNP in the case of vector fields. Inputs are the red arrows and the model extracts the whole
vector field.

The model consists of an encoder called **EquivDeepSet** and a decoder which is an equivariant neural network 
of the form outlined in CITE.

## Links

- We made use the library [E(2)-Steerable CNNs](https://github.com/QUVA-Lab/e2cnn)
provided for any group-related objects 
- We used PyTorch as a library for automatic differentation.

## How to use this repository

```shell
packagemanager install awesome-project
awesome-project start
awesome-project "Do something!"  # prints "Nah."
```

## Structure of the repository

## Acknowledgement 
This project was part my master's dissertation supervised by Professor
Yee Whye Teh (University of Oxford, Google DeepMind) and co-supervised by Michael Hutchinson
(University of Oxford).

## Contact

If you have any questions, feel free to contact me (peter.holderrieth@new.ox.ac.uk).
