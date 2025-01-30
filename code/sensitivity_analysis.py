# Sensitivity analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
# Apply PAWN PMF method to each grid cell
# Number of inputs:
M = X.shape[1]
# PAWN set-up
Nboot = 100
n = 10

max_dist_vals = np.empty((M, nloc_land))
max_dist_lbs = np.empty((M, nloc_land))
max_dist_ubs = np.empty((M, nloc_land))

for loc in range(nloc_land):
    print(loc)
    Y_loc = Y[:,loc]
    # Check number of decisions in location loc
    if len(np.unique(Y_loc)) == 1:
        print('Only one location in location ' + str(loc))
        # Fill in a row of NaNs
        max_dist_vals[:,loc] = np.repeat(np.nan, M)
        max_dist_lbs[:,loc] = np.repeat(np.nan, M)
        max_dist_ubs[:,loc] = np.repeat(np.nan, M)
        continue
    # Otherwise, run PAWN using PMFs
    max_dist_median, max_dist_mean, max_dist_max = \
        PAWN_pmf.pawn_pmf_indices(X_numeric, Y_loc, n = n, Nboot = Nboot)
    max_maxdist_mean, max_maxdist_lb, max_maxdist_ub = aggregate_boot(max_dist_max)
    max_dist_vals[:,loc] = max_maxdist_mean
    max_dist_lbs[:,loc] = max_maxdist_lb
    max_dist_ubs[:,loc] = max_maxdist_ub

# Save results
# np.save('./data/pawn_results/max_dist_vals_lhc200.npy', max_dist_vals)
# np.save('./data/pawn_results/max_dist_lbs_lhc200.npy', max_dist_lbs)
# np.save('./data/pawn_results/max_dist_ubs_lhc200.npy', max_dist_ubs)

######################################################################
# Plot map of sensitivity values

lon = data['lon']
lat = data['lat']

# Get the range of values:
np.nanmin(max_dist_vals), np.nanmax(max_dist_vals)

# Make plot with 3 rows x 4 columns
fig = plt.figure(figsize=(17,12))
ax1 = plt.subplot(3,4,1,projection=ccrs.PlateCarree())
ax2 = plt.subplot(3,4,2,projection=ccrs.PlateCarree())
ax3 = plt.subplot(3,4,3,projection=ccrs.PlateCarree())
ax4 = plt.subplot(3,4,4,projection=ccrs.PlateCarree())
ax5 = plt.subplot(3,4,5,projection=ccrs.PlateCarree())
ax6 = plt.subplot(3,4,6,projection=ccrs.PlateCarree())
ax7 = plt.subplot(3,4,7,projection=ccrs.PlateCarree())
ax8 = plt.subplot(3,4,8,projection=ccrs.PlateCarree())
ax9 = plt.subplot(3,4,9,projection=ccrs.PlateCarree())
ax10 = plt.subplot(3,4,10,projection=ccrs.PlateCarree())
ax11 = plt.subplot(3,4,11,projection=ccrs.PlateCarree())

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cp = ax1.scatter(lon,lat,c=max_dist_vals[0,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax1,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[0])
ax1.title.set_text('(a)')

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(lon,lat,c=max_dist_vals[1,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax2,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[1])
ax2.title.set_text('(b)')

ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
cp = ax3.scatter(lon,lat,c=max_dist_vals[2,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax3,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[2])
ax3.title.set_text('(c)')

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
cp = ax4.scatter(lon,lat,c=max_dist_vals[3,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax4,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[3])
ax4.title.set_text('(d)')

ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
cp = ax5.scatter(lon,lat,c=max_dist_vals[4,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax5,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[4])
ax5.title.set_text('(e)')

ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
cp = ax6.scatter(lon,lat,c=max_dist_vals[5,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax6,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[5])
ax6.title.set_text('(f)')

ax7.set_xlabel('Longitude')
ax7.set_ylabel('Latitude')
cp = ax7.scatter(lon,lat,c=max_dist_vals[6,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax7,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[6])
ax7.title.set_text('(g)')

ax8.set_xlabel('Longitude')
ax8.set_ylabel('Latitude')
cp = ax8.scatter(lon,lat,c=max_dist_vals[7,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax8,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[7])
ax8.title.set_text('(h)')

ax9.set_xlabel('Longitude')
ax9.set_ylabel('Latitude')
cp = ax9.scatter(lon,lat,c=max_dist_vals[8,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax9,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[8])
ax9.title.set_text('(i)')

ax10.set_xlabel('Longitude')
ax10.set_ylabel('Latitude')
cp = ax10.scatter(lon,lat,c=max_dist_vals[9,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax10,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[9])
ax10.title.set_text('(j)')

ax11.set_xlabel('Longitude')
ax11.set_ylabel('Latitude')
cp = ax11.scatter(lon,lat,c=max_dist_vals[10,:],vmin=0,vmax=0.5,s=5,cmap='viridis')
cbar = plt.colorbar(cp,ax=ax11,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[10])
ax11.title.set_text('(k)')

plt.show()