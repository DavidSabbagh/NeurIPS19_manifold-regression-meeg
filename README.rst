Regression with covariance matrices
===================================

This is the Python code for the NeurIPS 2019 article
**Manifold-regression to predict from MEG/EEG brain signals without source
modeling** 

Dependencies
------------

 - numpy >= 1.15
 - scipy >= 1.12
 - matplotlib >= 3.0
 - scikit-learn >= 0.20
 - pyriemann (https://github.com/alexandrebarachant/pyRiemann)

Libraries
---------

- **/library/generation.py** contains the functions to generate covariance matrices following the generative model of the paper

- **/library/wasserstein_tangent.py** implements the vectorization with the Wasserstein distance.
Vectorization using the geometric distance is performed using the PyRiemann package.

- **/library/utils.py** contains the other vectorization methods

Main scripts
-------------
The SNR analysis is obtained by running the script snr_experiment.py

The distance analysis is obtained by running the script distance_experiment.py

The individual mixing matrices is obtained by running the script
individual_noise_experiment.py

