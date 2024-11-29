# Uncertainty and sensitivity analysis of risk and decision

Code related to paper on U&SA of Bayesian Decision Analysis

## Data

Data should be placed in the folder `data`. There is a different folder per
+ Calibration method (Hazard): non (UKCP_raw), bias corrected (UKCP_BC), change factor (ChangeFactor)
+ Within each folder, files containing the output for all combinations of
    - Global Warming level (Hazard): current climate, 2oC of global warming, 4oC of global warming
    - UK Shared Socio-economic Pathway (Exposure): SSP1, SSP2, SSP5
    - Vulnerability function parameter 1 (Vulnerability): low, medium, high
    - Vulnerability function parameter 2: low, medium, high

In each case the data provided is
+ The 1000 samples from the Generalised Additive Model (GAM) fit to the expected annual impact at all grid cells over the UK. This is given as a NetCDF file –a common data format for climate information (talk to Dan about this if you have any questions)
+ The ‘optimal decision’ in each location, identified using the Bayesian Decision Analysis framework, for a given combination of inputs including the ‘cost per day of work lost’ and ‘relative importance of cost vs meeting company objectives’. This is given as a csv file with columns for longitude, latitude and the optimal decision

## Code

Files for the following purposes:

+ `climada_env_2022.yaml`: Python environment
+ `py_functions.py`: Python functions to be sourced in other scripts
+ `default_dec_attr.py`: Code pertaining to the default decision attributes
