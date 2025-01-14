import numpy as np
import pandas as pd
from safepython.PAWN import pawn_split_sample
from safepython.util import empiricalcdf

# Define risk inputs
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = [53.78, 54.5, 55.79]
vuln2_opts = [-4.597, -4.1, -3.804]

X_risk = np.array(np.meshgrid(ssp_opts, warming_opts, calibration_opts, vuln1_opts, vuln2_opts)).T.reshape(-1,5)
# X_risk.shape # (162, 5)

# Need to make a numeric version of X for PAWN
X_risk_numeric = np.array(np.meshgrid([1,2,5], [2,4], [1,2,3], vuln1_opts, vuln2_opts)).T.reshape(-1,5)

# Load in Latin hypercube samples
lhc_200 = np.loadtxt('./data/lat_hyp_samples_200.csv', delimiter=',')
# lhc_200.shape # (200, 6)

# X needs to be of shape (32400, 11)
# Repeat every element of X_risk 200 times
X_risk_repeated = np.repeat(X_risk, lhc_200.shape[0], axis=0)
X_risk_numeric_repeated = np.repeat(X_risk_numeric, lhc_200.shape[0], axis=0)
# X_risk_repeated.shape # (32400, 5)

# Tile lhc_200 162 times (repeating the entire array 162 times)
lhc_200_tiled = np.tile(lhc_200, (X_risk_numeric.shape[0], 1))
# lhc_200_tiled.shape # (32400, 6)

# Combine to get the final array of shape (32400, 11)
X = np.hstack((X_risk_repeated, lhc_200_tiled))
X_numeric = np.hstack((X_risk_numeric_repeated, lhc_200_tiled))
# X.shape # (32400, 11)

# calculate output (optimal decision in each land location)
nloc_land = 1711
# Make Y with 32400 rows and 1711 columns
Y = np.empty(X.shape[0]*nloc_land).reshape(X.shape[0],nloc_land)

for i in range(X.shape[0]):
    risk_input_string = 'ssp'+X[i,0]+'_'+X[i,1]+'_'+X[i,2]+'_v1_'+X[i,3]+'_v2_'+X[i,4]
    dec_input_string = '_cpd_'+X[i,5]+'_fweight_'+X[i,10]+'_d2_'+str(X[i,6])+'_'+str(X[i,7])+'_6.0_d3_'+str(X[i,8])+'_'+str(X[i,9])+'_4.0'
    data = pd.read_csv('./data/decision_files_200/OptimalDecision_'+risk_input_string+dec_input_string+'.csv')
    Y[i,:] = data['optimal_decision']

# Make Y for a single location (London)
Y_loc = Y[:,241]

# Run through the original PAWN code
Nboot=1
dummy=False
# output_condition=allrange

###########################################################################
# Check inputs and split the input sample
###########################################################################

# Using default n value for pawn_split_sample
# n = 10

YY, xc, NC, n_eff, Xk, XX = pawn_split_sample(X_numeric, Y_loc, n=10) # this function
# checks inputs X, Y and n

Nx = X.shape # (32400, 11)
N = Nx[0] # 32400
M = Nx[1] # 11

###########################################################################
# Check other optional inputs
###########################################################################

if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
    raise ValueError('"Nboot" must be scalar and integer.')
if Nboot < 1:
    raise ValueError('"Nboot" must be >=1.')
if not isinstance(dummy, bool):
    raise ValueError('"dummy" must be scalar and boolean.')
# if not callable(output_condition):
    # raise ValueError('"output_condition" must be a function.')

###########################################################################
# Compute indices
###########################################################################

# Set points at which the CDFs will be evaluated:
YF = np.unique(Y) # YF is an array containing 1, 2, and 3

# Initialize sensitivity indices
KS_median = np.nan * np.ones((Nboot, M))
KS_mean = np.nan * np.ones((Nboot, M))
KS_max = np.nan * np.ones((Nboot, M))
if dummy: # Calculate index for the dummy input
    KS_dummy = np.nan * np.ones((Nboot, ))

# Compute conditional CDFs
# (bootstrapping is not used to assess conditional CDFs):
FC = [np.nan] * M
for i in range(M): # loop over inputs
    FC[i] = [np.nan] * n_eff[i] # n_eff is the number of conditioning intervals used for each input
    for k in range(n_eff[i]): # loop over conditioning intervals
        FC[i][k] = empiricalcdf(YY[i][k], YF)
# FC is a list of length 11 (i)
# Each element is a list containing n_eff numpy arrays
# Each numpy array has three elements
