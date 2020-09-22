# Wearable Data Affective State Classifier

In this project I have built a multi-class emotional affect classifier in Python using a 3-layer long short-term memory (LSTM) recurrent neural network trained on wearable data.  The classifier predicts three states of affect--baseline, stress and amusement using the wearable data.  To accomplish this, I downloaded and used a publicly available dataset for WEarable Stress and Affect Detection (WESAD), available here:

https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29#

WESAD is a multimodal dataset that includes physiological and motion data along with self-reports of the subjects affect.  The following physiological sensor modalities were measured from the wrist and/or chest of each subject: blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and three-axis acceleration. Self-reports of the subjects affect were obtained using several established questionnaires.

Here in this project I use a subset of these modalities--accelerometer data, temperature data (skin temperature), and electrodermal activity to classify affective state.

## File Organization and Function 

  - Proof_Of_Concept.ipynb - ipython notebook with script to create the WESAD_data object that runs the data input, processing, visualization and model creation, fitting and testing.  
    - WESAD_data.py - python module with WESAD_data class definition, along with accompanying attributes and methods to perform data input and processing.  

## Environment and Instructions

You must first ensure the data is downloaded correctly.

There are two paths that need to be set correctly to run the notebook and python module:
  - path to the 'WESAD_dataset.py' python module.  This is set on lines 8 of 'Proof of Concept.ipynb.'
  - a path to the WESAD dataset. This is set one line 34 of 'WESAD_data.py'

## Features

The features of acceleration used were:

    mean for each axis; summed over all axes
    STD for each axis; summed over all axes
    Peak frequency for x
    Peak frequency for Y
    Peak frequency for z

The features of temperature used were:

    Min value
    max value
    Dynamic Range
    Mean
    STD

The features of electrodermal acitivty used were:

    Min value
    max value
    Dynamic Range
    Mean
    STD

## Results

Multi-class classification of this dataset has been done using other algorithms such as decision trees, random forests, Adaboost DT, linear discriminant analysis and k-nearest neighbors (https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf).  The previous best reported multi-class classification performance using the ADABoost DT algorithm was 80% accuracy.  Using a three-layer LSTM neural network to do multi-class classification on this dataset, my solution improved the performance on the multi-class classification to 92% accuracy using only a subset of features used in the referenced paper.  This means my model improved accuracy by approximately 12% over the current benchmark with less features.

## LSTM justifications and considerations
The long short-term memory neural network is a type of neural network that in addition to the regular inputs at each node also have a recurrent input that allows the network to take time and sequence into account, effectively giving it a temporal dimension.   

Here I've used an LSTM for two reasons:

1) The wearable data is temporal and LSTMs specialize in recognizing patterns in sequences of data.  
2) Though many other algorithms were tested on this datset, there were no previous reports of an LSTM being tested. 

Advantages:
- Handle noise and non-stationarity well
- LSTMs generalize well (validation accuracy is higher than training accuracy)
- LSTMs work well over many hyperparameters, i.e. there is no need for parameter tuning
- With constant error back propagation through memory cells, LSTMs can bridge very long time lags in the data
- LSTMs update complexity per weight and time step is essentially O(1)

Disadvantages: 
- LSTMs take longer to train than other methods
- LSTMs take more memory to train (because of higher number of parameters to fit)
- LSTMs are easy to overfit
- Difficult to implement dropout 
- Sensitive to random weight initializations

## Next step future directions 

The next step in this project is creating a dashboard that would present useful summary information such as percentages of time spent in each affect, ratios of time spent in affects and temporal patterns of affect across days and weeks.  The dashboard would allow both individual and group summaries. 

