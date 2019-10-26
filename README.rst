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

- **library/preprocessing.py** contains the code used to preprocess raw data from CamCAN

- **/library/spfiltering.py** contains the functions to implement spatial filtering of the covariance matrices

- **/library/featuring.py** contains all the functions to vectorize the covariance matrices 

- **library/simuls**: contains the function to  generate covariance matrices following the generative model of the paper

- **/library/utils.py** contains the other vectorization methods

Main scripts
-------------

- **nips_simuls_compute_** are the 3 scripts used for the 3 simulations of the paper
 
- **nips_simuls_plot_** are the corresponding plotting scripts (in R)
