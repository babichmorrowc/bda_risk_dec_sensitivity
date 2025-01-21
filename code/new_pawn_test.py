import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from safepython.PAWN import pawn_split_sample, pawn_ks
from safepython.util import empiricalcdf, allrange

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
# For the function pawn_indices
# https://github.com/SAFEtoolbox/SAFE-python/blob/main/src/safepython/PAWN.py#L213
# Setting the default parameters
Nboot=1
dummy=False
output_condition=allrange
par=[]

###########################################################################
# Check inputs and split the input sample
###########################################################################

# For detail on the code for pawn_split_sample:
# https://github.com/SAFEtoolbox/SAFE-python/blob/main/src/safepython/PAWN.py#L37
# Using the default n=10: number of conditioning intervals
YY, xc, NC, n_eff, Xk, XX = pawn_split_sample(X_numeric, Y_loc, n=10) # this function
# checks inputs X, Y and n
# YY: list of length M=11 
## Each element is a list of n_eff[i] arrays 
## that can be used to assess n_eff[i] conditional output distributions wrt the i-th input variable
## YY[i][k] fixes the i-th input to its k-th conditioning interval
# xc: list of length M=11
## Contains the mean value of each input variable over each conditioning interval
# NC: list of length M=11
## Each element is an array of n_eff[i] values
## representing the number of data points in each conditioning interval for that input
# n_eff: list of length M=11
## Contains the number of conditioning intervals for each input
# Xk: list of length M=11
## Each element is an array of length n_eff[i] + 1
## Bounds of Xi over each conditioning interval
# XX: list of length M=11
## Each element is a list of n_eff[i] subsamples for the i-th input parameter
## XX[i][k] fixes the i-th input to its k-th conditioning interval (and the other inputs vary freely)

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
if not callable(output_condition):
    raise ValueError('"output_condition" must be a function.')

###########################################################################
# Compute indices
###########################################################################

# Set points at which the CDFs will be evaluated:
YF = np.unique(Y_loc) # YF is an array containing 1, 2, and 3

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
        # This gets the empirical cdf of YY[i][k] at YF
        # i.e. at the three unique output decision values
# FC is a list of length 11 (i)
# Each element is a list containing n_eff[i] numpy arrays
# Each numpy array has three elements

################################################################################
# # Go through empiricalcdf code once to understand what that's doing
# x = YY[0][0].flatten()
# xi = YF.flatten()

# N = len(x) # n_eff[i]
# F = np.linspace(1, N, N)/N
# # linspace gives N evenly spaced numbers over an interval between 1 and N (then divide by N)
# # So F goes from 1/N to 1
# # length is N

# # Remove any multiple occurance of 'x'
# # and set F(x) to the upper value (recall that F(x) is the percentage of
# # samples whose value is lower than *or equal to* x!)
# # We save the indices of the last occurence of each element in the vector 'x',
# # when 'x' is sorted in ascending order. 

# x = np.sort(x) # sort x in ascending order
# x_u = np.unique(x) # get unique values of x (in this case [1,2,3])
# iu = np.array([np.where(x_u[ii]==x)[0][-1] for ii in range(len(x_u))])
# #extract indices of the last occurence of each unique value in x

# F = F[iu] # subset F down to the indices of the last occurrence of each unique value
# N = len(F) # 3

# # Interpolate the empirical CDF at 'xi':
# Fi = np.ones((len(xi),)) # array of length 3 that is all 1's

# for j in range(N-1, -1, -1): # counting down from N-1=2 to 0
#     print(j)
#     Fi[xi[:] <= x_u[j]] = F[j]
#     print(Fi)
#     # For all indices of Fi where xi is less than or equal to the output value
#     # (going from largest output value to smallest)
#     # set Fi equal to the index of the last occurrence of that output value

# Fi[xi < x_u[0]] = 0
# Fi
################################################################################

# Now I want to instead compute the conditional PMF
# of YY[i][k] at YF
fC = [np.nan] * M
for i in range(M): # loop over inputs
    fC[i] = [np.nan] * n_eff[i] # n_eff is the number of conditioning intervals used for each input
    for k in range(n_eff[i]): # loop over conditioning intervals
        fC[i][k] = np.unique(YY[i][k], return_counts=True)[1]/len(YY[i][k])
        # This gets the empirical pmf of YY[i][k] at YF
        # i.e. at the three unique output decision values
fC

#################################################################################
# Initialize unconditional CDFs:
FU = [np.nan] * M

# M unconditional CDFs are computed (one for each input), so that for
# each input the conditional and unconditional CDFs are computed using the
# same number of data points (when the number of conditioning intervals
# n_eff[i] varies across the inputs, so does the shape of the conditional
# outputs YY[i]).

 # Determine the sample size for the unconditional output bootsize:
bootsize = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
# bootsize is equal to the sample size of the conditional outputs NC, or
# its  minimum value across the conditioning intervals when the sample size
# varies across conditioning intervals as may happen when values of an
# input are repeated several times (more details on this in the Note in the
# help of the function).

# To reduce the computational time (the calculation of empirical CDF is
# costly), the unconditional CDF is computed only once for all inputs that
# have the same value of bootsize[i].
bootsize_unique = np.unique(bootsize)
N_compute = len(bootsize_unique)  # number of unconditional CDFs that will
# be computed for each bootstrap resample
# Either of size 10800 or 3240

# NOTE: I'm skipping the dummy part here

# Compute sensitivity indices with bootstrapping
for b in range(Nboot): # number of bootstrap resample

    # Compute empirical unconditional CDFs
    for kk in range(N_compute): # loop over the sizes of the unconditional output

        # Bootstrap resampling (Extract an unconditional sample of size
        # bootsize_unique[kk] by drawing data points from the full sample Y
        # without replacement
        idx_bootstrap = np.random.choice(np.arange(0, N, 1), # evenly spaced values between 0 and N, spaced by 1
                                         size=(bootsize_unique[kk], ),
                                         replace='False')
        # Compute unconditional CDF:
        FUkk = empiricalcdf(Y_loc[idx_bootstrap], YF)
        # Associate the FUkk to all inputs that require an unconditional
        # output of size bootsize_unique[kk]:
        idx_input = np.where([i == bootsize_unique[kk] for i in bootsize])[0]
        for i in range(len(idx_input)):
            FU[idx_input[i]] = FUkk
    # Compute KS statistic between conditional and unconditional CDFs:
    KS_all = pawn_ks(YF, FU, FC, output_condition, par)
    # KS_all is a list (M elements) and contains the value of the KS for
    # for each input and each conditioning interval. KS[i] contains values
    # for the i-th input and the n_eff[i] conditioning intervals, and it
    # is a numpy.ndarray of shape (n_eff[i], ).
    #  Take a statistic of KS across the conditioning intervals:
    KS_median[b, :] = np.array([np.median(j) for j in KS_all])  # shape (M,)
    KS_mean[b, :] = np.array([np.mean(j) for j in KS_all])  # shape (M,)
    KS_max[b, :] = np.array([np.max(j) for j in KS_all])  # shape (M,)

################################################################################

# Now I want to instead compute the unconditional PMF
# of YY[i][k] at YF


# Initialize unconditional PMFs:
fU = [np.nan] * M
bootsize = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
bootsize_unique = np.unique(bootsize)
N_compute = len(bootsize_unique)

# NOTE: I'm skipping the dummy part here

# Compute sensitivity indices with bootstrapping
for b in range(Nboot): # number of bootstrap resample

    # Compute empirical unconditional CDFs
    for kk in range(N_compute): # loop over the sizes of the unconditional output

        # Bootstrap resampling (Extract an unconditional sample of size
        # bootsize_unique[kk] by drawing data points from the full sample Y
        # without replacement
        idx_bootstrap = np.random.choice(np.arange(0, N, 1), # evenly spaced values between 0 and N, spaced by 1
                                         size=(bootsize_unique[kk], ),
                                         replace='False')
        # Compute unconditional CDF:
        fUkk = np.unique(Y_loc[idx_bootstrap], return_counts = True)[1]/len(Y_loc[idx_bootstrap])
        # Associate the fUkk to all inputs that require an unconditional
        # output of size bootsize_unique[kk]:
        idx_input = np.where([i == bootsize_unique[kk] for i in bootsize])[0]
        for i in range(len(idx_input)):
            fU[idx_input[i]] = fUkk
    # Compute KS statistic between conditional and unconditional CDFs:
    KS_all = pawn_ks(YF, fU, FC, output_condition, par)
    # KS_all is a list (M elements) and contains the value of the KS for
    # for each input and each conditioning interval. KS[i] contains values
    # for the i-th input and the n_eff[i] conditioning intervals, and it
    # is a numpy.ndarray of shape (n_eff[i], ).
    #  Take a statistic of KS across the conditioning intervals:
    KS_median[b, :] = np.array([np.median(j) for j in KS_all])  # shape (M,)
    KS_mean[b, :] = np.array([np.mean(j) for j in KS_all])  # shape (M,)
    KS_max[b, :] = np.array([np.max(j) for j in KS_all])  # shape (M,)

#################################################################################

if Nboot == 1:
    KS_median = KS_median.flatten()
    KS_mean = KS_mean.flatten()
    KS_max = KS_max.flatten()