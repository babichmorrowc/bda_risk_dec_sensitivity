# Sensitivity analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy.crs as ccrs
from safepython import PAWN
import safepython.PAWN_pmf as PAWN_pmf
from safepython.util import aggregate_boot  # function to aggregate the bootstrap results

###################################################################
# Set up inputs and outputs
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

###################################################################
# SENSITIVITY OF RISK
###################################################################

# Calculate mean and upper and lower credible intervals from GAM samples
nloc = 110*83
Y_mean = np.empty((X_risk.shape[0],nloc))
# Y_lower = np.empty(X_risk.shape[0])
# Y_upper = np.empty(X_risk.shape[0])

for i in range(X_risk.shape[0]):
    data = Dataset('./data/'+X_risk[i,2]+
                   '/GAMsamples_expected_annual_impact_data_'+X_risk[i,2]+
                   '_WL'+X_risk[i,1]+
                   '_SSP'+X_risk[i,0]+
                   '_vp1='+X_risk[i,3]+
                   '_vp2='+X_risk[i,4]+'.nc')
    allEAI = np.array(data.variables['sim_annual_impact'])
    allEAI[np.where(allEAI > 9e30)] = 0
    allEAI = 10**allEAI - 1 # 110 x 83 x 1000
    # get average EAI in each location
    meanEAI = np.mean(allEAI, axis=2)
    meanEAI = meanEAI.reshape(110*83)
    Y_mean[i,:] = meanEAI
    # aggEAI = np.nansum(allEAI,axis=(0,1)) # 1000
    # Y_mean[i] = np.mean(aggEAI)
    # Y_lower[i] = np.quantile(aggEAI,0.025)
    # Y_upper[i] = np.quantile(aggEAI,0.975)

###################################################################
# SENSITIVITY OF OUTPUT DECISION
###################################################################

# Expand X to include decision-related parameters

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

# X labels
X_labels = ['SSP',
            'Warming level',
            'Calibration method',
            'Vulnerability parameter 1',
            'Vulnerability parameter 2',
            'Cost per day of work lost',
            'Annual cost per person of d2',
            'Effectiveness of d2',
            'Annual cost per person of d3',
            'Effectiveness of d3',
            'Relative importance of financial']

# calculate output (optimal decision in each land location)
nloc_land = 1711
# Make Y with 32400 rows and 1711 columns
Y = np.empty(X.shape[0]*nloc_land).reshape(X.shape[0],nloc_land)

for i in range(X.shape[0]):
    risk_input_string = 'ssp'+X[i,0]+'_'+X[i,1]+'_'+X[i,2]+'_v1_'+X[i,3]+'_v2_'+X[i,4]
    dec_input_string = '_cpd_'+X[i,5]+'_fweight_'+X[i,10]+'_d2_'+str(X[i,6])+'_'+str(X[i,7])+'_6.0_d3_'+str(X[i,8])+'_'+str(X[i,9])+'_4.0'
    data = pd.read_csv('./data/decision_files_200/OptimalDecision_'+risk_input_string+dec_input_string+'.csv')
    Y[i,:] = data['optimal_decision']

################################################################
# # Apply PAWN PMF method to each grid cell
# # Number of inputs:
# M = X.shape[1]
# # PAWN set-up
# Nboot = 100
# n = 10

# max_dist_vals = np.empty((M, nloc_land))
# max_dist_lbs = np.empty((M, nloc_land))
# max_dist_ubs = np.empty((M, nloc_land))

# for loc in range(nloc_land):
#     print(loc)
#     Y_loc = Y[:,loc]
#     # Check number of decisions in location loc
#     if len(np.unique(Y_loc)) == 1:
#         print('Only one location in location ' + str(loc))
#         # Fill in a row of NaNs
#         max_dist_vals[:,loc] = np.repeat(np.nan, M)
#         max_dist_lbs[:,loc] = np.repeat(np.nan, M)
#         max_dist_ubs[:,loc] = np.repeat(np.nan, M)
#         continue
#     # Otherwise, run PAWN using PMFs
#     max_dist_median, max_dist_mean, max_dist_max = \
#         PAWN_pmf.pawn_pmf_indices(X_numeric, Y_loc, n = n, Nboot = Nboot)
#     max_maxdist_mean, max_maxdist_lb, max_maxdist_ub = aggregate_boot(max_dist_max)
#     max_dist_vals[:,loc] = max_maxdist_mean
#     max_dist_lbs[:,loc] = max_maxdist_lb
#     max_dist_ubs[:,loc] = max_maxdist_ub

# Save results
# np.save('./data/pawn_results/max_dist_vals_lhc200.npy', max_dist_vals)
# np.save('./data/pawn_results/max_dist_lbs_lhc200.npy', max_dist_lbs)
# np.save('./data/pawn_results/max_dist_ubs_lhc200.npy', max_dist_ubs)

max_dist_vals = np.load('./data/pawn_results/max_dist_vals_lhc200.npy')

# Get the range of values:
np.nanmin(max_dist_vals), np.nanmax(max_dist_vals)

# max_dist_vals sensitivity of SSP ranked from high to low
# removing the nan values
# np.argsort(max_dist_vals[0,:])[::-1][~np.isnan(max_dist_vals[0,np.argsort(max_dist_vals[0,:])[::-1]])]
# import os
# os.chdir("./code")
# from plotting_functions import *
# plot_index(1445)

