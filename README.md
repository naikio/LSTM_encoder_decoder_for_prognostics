# LSTM Encoder-Decoder for prognostics
> WORK IN PROGRESS - this work still requires much further testing and review

> This work is a loose implementation of the work described in this paper: [Multi-Sensor Prognostics using an Unsupervised Health Index based on LSTM Encoder-Decoder](https://arxiv.org/abs/1608.06154)

The main idea is to build an LSTM encoder-decoder architecture that is going to be trained on time-series 
that refers to the healthy status of the machine.

The reconstruction errors will be then used to transform the Reconstruction Error curves into Health Index curves.

With all the Health-Index curves computed from the training set, a linear regressor is then trained.

At test-time it is possible to make a prediction of Remaining Useful Life (RUL), by using the input points and the regressor only. 
## Features

The dataset used here is the CMAPSS Turbofan Engine Simulation Dataset. It can be downloaded here: https://c3.nasa.gov/dashlink/resources/139/

A brief description of the main scripts:

* cmaps_dataset.py - contains useful functions to handle the dataset
* train.py - train the LSTM
* calc_reco_erros.py - inference-step for the LSTM to compute all the reconstruction errors of the series in the training set
* linear_regression.py - build linear regression from the previous step
* TODO : take test series and make predictions


### Testing

- - 
### Requirements

* Tensorflow, numpy
* Sci-kit learn (for PCA)
* csv
* matplotlib


## Contributing

