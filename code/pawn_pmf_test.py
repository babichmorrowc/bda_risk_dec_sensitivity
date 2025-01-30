# Test out the PAWN PMF function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# Test the pawn_pmf_indices function
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
test_median

# Test out the PAWN pmf function with bootstrapping
boot_test_median, boot_test_mean, boot_test_max = PAWN_pmf.pawn_pmf_indices(X = X_numeric, Y = Y_loc, n = 10, Nboot=50)
boot_test_median

#########################################################################
# Test out PMF plotting
YF, fU, fC, xc = PAWN_pmf.pawn_plot_pmf(X = X_numeric, Y = Y_loc, n_col=5, n = 10, cbar=True)
plt.show()

# Test out KS plotting
PAWN.pawn_plot_ks(YF, fU, fC, xc)
plt.show()