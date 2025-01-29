# Test out the PAWN PMF function
import numpy as np
import pandas as pd
from safepython import PAWN_pmf
from safepython import PAWN

# Create a vector Y of length 100 with values 1, 2, 3
# Proportions are 0.1, 0.2, 0.7
Y = np.array(np.repeat([1, 2, 3], [10, 20, 70]))
# Randomly shuffle the vector
np.random.shuffle(Y)

# Test whether the np.unique part of my function is working as expected
# np.unique(Y, return_counts=True)[1] / len(Y)
# all good!
# Works even when shuffled

# What if I have a subset of Y that only contains 1s and 3s?
Y_sub = np.array(np.repeat([1,3], [2,8]))
np.random.shuffle(Y_sub)

# I'd like to end up with [0.2, 0.0, 0.8]
# np.unique(Y_sub, return_counts=True)[1] / len(Y_sub)
# np.sum(Y_sub == 1)
Y_sub_unique, Y_sub_counts = np.unique(Y_sub, return_counts = True)
Y_sub_count_dict = dict(zip(Y_sub_unique, Y_sub_counts))
counts_final = np.array([Y_sub_count_dict.get(val, 0) for val in np.unique(Y)])
prop_final = counts_final / len(Y_sub)

################################################################################
# Test the function
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

# Test out the PAWN pmf function
test_median, test_mean, test_max = PAWN_pmf.pawn_pmf_indices(X = X_numeric, Y = Y_loc, n = 10)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/aw23877/Documents/bda_sensitivity_paper/SAFE-python/src/safepython/PAWN_pmf.py", line 329, in pawn_pmf_indices
#     KS_all = pawn_ks(YF, fU, fC, output_condition, par)
#   File "/home/aw23877/Documents/bda_sensitivity_paper/SAFE-python/src/safepython/PAWN.py", line 613, in pawn_ks
#     KS[i][k] = np.max(abs(FU[i][idx] - FC[i][k][idx]))
# IndexError: boolean index did not match indexed array along dimension 0; dimension is 2 but corresponding boolean dimension is 3

##################################################################################
# Let's try line-by-line
X = X_numeric
Y = Y_loc
Nboot = 1
output_conditions = PAWN.allrange
par = []

YY, xc, NC, n_eff, Xk, XX = PAWN.pawn_split_sample(X, Y, n = 10) # this function
# checks inputs X, Y and n

Nx = X.shape
N = Nx[0]
M = Nx[1]

# Set points at which the PMFs will be evaluated:
YF = np.unique(Y)

 # Initialize sensitivity indices
KS_median = np.nan * np.ones((Nboot, M))
KS_mean = np.nan * np.ones((Nboot, M))
KS_max = np.nan * np.ones((Nboot, M))

 # Compute conditional PMFs
# (bootstrapping is not used to assess conditional PMFs):
fC = [np.nan] * M
for i in range(M): # loop over inputs
    fC[i] = [np.nan] * n_eff[i]
    for k in range(n_eff[i]): # loop over conditioning intervals
        fC[i][k] = np.unique(YY[i][k], return_counts=True)[1]/len(YY[i][k])

 # Initialize unconditional PMFs:
fU = [np.nan] * M

 # M unconditional PMFs are computed (one for each input), so that for
# each input the conditional and unconditional PMFs are computed using the
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

 # To reduce the computational time (the calculation of empirical PMF is
# costly), the unconditional PMF is computed only once for all inputs that
# have the same value of bootsize[i].
bootsize_unique = np.unique(bootsize)
N_compute = len(bootsize_unique)

# Compute sensitivity indices with bootstrapping
for b in range(Nboot): # number of bootstrap resample

     # Compute empirical unconditional PMFs
    for kk in range(N_compute): # loop over the sizes of the unconditional output

         # Bootstrap resapling (Extract an unconditional sample of size
        # bootsize_unique[kk] by drawing data points from the full sample Y
        # without replacement
        idx_bootstrap = np.random.choice(np.arange(0, N, 1),
                                         size=(bootsize_unique[kk], ),
                                         replace='False')
        # Compute unconditional PMF:
        fUkk = np.unique(Y[idx_bootstrap], return_counts=True)[1]/len(Y[idx_bootstrap])
        # Associate the fUkk to all inputs that require an unconditional
        # output of size bootsize_unique[kk]:
        idx_input = np.where([i == bootsize_unique[kk] for i in bootsize])[0]
        for i in range(len(idx_input)):
            fU[idx_input[i]] = fUkk

     # Compute KS statistic between conditional and unconditional CDFs:
    KS_all = PAWN.pawn_ks(YF, fU, fC, PAWN.allrange, par)
    # KS_all is a list (M elements) and contains the value of the KS for
    # for each input and each conditioning interval. KS[i] contains values
    # for the i-th input and the n_eff[i] conditioning intervals, and it
    # is a numpy.ndarray of shape (n_eff[i], ).

     #  Take a statistic of KS across the conditioning intervals:
    KS_median[b, :] = np.array([np.median(j) for j in KS_all])  # shape (M,)
    KS_mean[b, :] = np.array([np.mean(j) for j in KS_all])  # shape (M,)
    KS_max[b, :] = np.array([np.max(j) for j in KS_all])  # shape (M,)
