import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import cartopy.crs as ccrs
import safepython.plot_functions as pf 
from netCDF4 import Dataset

###########################################################################
# Set-up
# Set up range of risk options to vary
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = ["53.78", "54.5", "55.79"]
vuln2_opts = ["-4.597", "-4.1", "-3.804"]

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
X_labels_risk = X_labels[0:5]
# Short versions
X_labels_short = ['SSP',
                  'WARM',
                  'CAL',
                  'VULN1',
                  'VULN2',
                  'COST1',
                  'COST2',
                  'COST3',
                  'EFF1',
                  'EFF2',
                  'WEIGHT']
X_labels_risk_short = X_labels_short[0:5]

# Color scales
# Continuous

# Discrete, sequential
# dis_seq_cols = ['#FEE6CE','#FDAE6B','#E6550D']
# dis_seq_cols = ['#ffeda0','#FDAE6B','#E6550D']
dis_seq_cols = ['#E9C46A', '#F4A261', '#E76F51']
dis_seq = ListedColormap(dis_seq_cols)
# Discrete, nonsequential
dis_non_cols = ['#40476D', '#5296A5']

# Plotting set-up for uncertainty maps
levels = [1,2,3,4]
dis_seq_norm = BoundaryNorm(levels, dis_seq.N)
legend_labs = ['1 optimal decision', '2 optimal decisions', '3 optimal decisions']

###########################################################################
# FIGURE 2A
# Standard deviation of risk in each grid cell
# Roughly following Laura's code here:
# https://github.com/babichmorrowc/compass_miniproject/blob/main/code/orig_laura_code/plots_for_paper_python.py#L471
X_risk = np.array(np.meshgrid(ssp_opts, warming_opts, calibration_opts, vuln1_opts, vuln2_opts)).T.reshape(-1,5)

# can't make full output matrix - memory issues, so calculate combined SD following http://www.obg.cuhk.edu.hk/ResearchSupport/StatTools/CombineMeansSDs_Pgm.php#:~:text=The%20Standard%20Error%20of%20the,sizes%20from%20all%20the%20groups.
txx = np.empty(110*83*X_risk.shape[0]).reshape(110,83,X_risk.shape[0])
tx = np.empty(110*83*X_risk.shape[0]).reshape(110,83,X_risk.shape[0])

# Pretty slow -- can I jit this somehow?
for i in range(X_risk.shape[0]):
    data = Dataset('./data/'+X_risk[i,2]+
                    '/GAMsamples_expected_annual_impact_data_'+X_risk[i,2]+
                    '_WL'+X_risk[i,1]+
                    '_SSP'+X_risk[i,0]+
                    '_vp1='+X_risk[i,3]+'_vp2='+X_risk[i,4]+'.nc')
    allEAI = np.array(data.variables['sim_annual_impact'])
    # land locations have EAI < 9e30
    allEAI[np.where(allEAI > 9e30)] = 0
    allEAI = 10**allEAI - 1
    # get sum of squared eai values across all gam samples
    txx[:,:,i] = np.apply_along_axis(lambda x : sum(x**2), 2, allEAI)
    # get sum of eai values across all gam samples
    tx[:, :, i] = np.apply_along_axis(lambda x: sum(x), 2, allEAI)

txx_all1 = np.apply_along_axis(lambda x : sum(x), 2, txx)
tx_all1 = np.apply_along_axis(lambda x : sum(x), 2, tx)

SDall = np.empty(110*83).reshape(110,83)
for i in range(110):
    for j in range(83):
        SDall[i,j] = np.sqrt((txx_all1[i,j] - tx_all1[i,j]**2/(X_risk.shape[0]*1000))/(X_risk.shape[0]*1000 -1))
nloc = 110*83
SDall = SDall.reshape(nloc)

lon = np.array(data.variables['exposure_longitude'])
lon = lon.reshape(nloc)
lat = np.array(data.variables['exposure_latitude'])
lat = lat.reshape(nloc)
log_norm = LogNorm(vmin=1.0e1, vmax=2.0e6)

###########################################################################
# FIGURE 2B
# Varying risk inputs ONLY
# Plot number of optimal decisions per cell

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = './data/decision_files_jit/'
file_list = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
n_files = len(file_list)

nloc_land = 1711
Y_risk = np.empty(n_files*nloc_land).reshape(n_files,nloc_land)

# Loop over files in the folder
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    Y_risk[i, :] = data['optimal_decision']

noptions_risk = np.empty(nloc_land)
for i in range(1711):
    noptions_risk[i] = len(np.unique(Y_risk[:,i]))
# np.unique(noptions_risk, return_counts = True)
noptions_risk = noptions_risk.astype(int)

################################################################################################
# FIGURE 2

# Plot standard deviation of risk
fig, axs = plt.subplots(1, 2, figsize=(18,15), layout="constrained", subplot_kw={'projection': ccrs.PlateCarree()})
# plt.figure(figsize=(18,15))
# ax1 = plt.subplot(1,2,1,projection=ccrs.PlateCarree())
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
cp = axs[0].scatter(lon,lat,c=SDall,
                 norm=log_norm,
                 s=11,
                 cmap='Purples')
cbar = plt.colorbar(cp,ax=axs[0],shrink=0.5)
cbar.set_label('Standard deviation of Risk')
axs[0].set_title('(a)')
axs[0].coastlines()

# Plot number of decisions optimal in each cell
lon_land = data['lon']
lat_land = data['lat']
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
scatter = axs[1].scatter(lon_land,lat_land,c=noptions_risk,s=12,cmap=dis_seq, norm=dis_seq_norm)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=legend_labs[i], 
           markersize=10, markerfacecolor=dis_seq_cols[i])
    for i in range(len(legend_labs))
]
axs[1].legend(handles=legend_elements)
axs[1].set_title('(b)')
axs[1].coastlines()

# Make sure plot (a) has the same limits
axs[0].set_xlim(axs[1].get_xlim())
axs[0].set_ylim(axs[1].get_ylim())
plt.show()

# # Plot percentage of time that each decision was optimal in a cell

# List of all possible decisions
# decision_options = np.unique(Y_risk, return_counts = True)

# # Calculate frequencies of each decision per location
# column_frequencies = {value: [] for value in decision_options[0]}
# num_rows = Y.shape[0]

# for col_index in range(Y.shape[1]):
#     unique_values, value_counts = np.unique(Y[:, col_index], return_counts=True)
#     # Create a dictionary to hold frequency counts for current column
#     freq_dict = dict(zip(unique_values, value_counts))
#     # Calculate frequencies for each decision option
#     for value in decision_options[0]:
#         if value in freq_dict:
#             frequency = freq_dict[value] / num_rows
#         else:
#             frequency = 0.0  # Value not present, set frequency to 0
        
#         column_frequencies[value].append(frequency)

# frequencies_df = pd.DataFrame(column_frequencies)

# fig = plt.figure(figsize=(18,9))
# ax1 = plt.subplot(1,3,1,projection=ccrs.PlateCarree())
# ax2 = plt.subplot(1,3,2,projection=ccrs.PlateCarree())
# ax3 = plt.subplot(1,3,3,projection=ccrs.PlateCarree())
# lon_land = data['lon']
# lat_land = data['lat']

# # Percent of time that decision 1 is optimal
# ax1.set_xlabel('Longitude')
# ax1.set_ylabel('Latitude')
# scatter = ax1.scatter(lon_land,lat_land,c=frequencies_df[1],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax1,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax1.coastlines()
# ax1.title.set_text('Decision 1: % samples optimal')

# # Percent of time that decision 2 is optimal
# ax2.set_xlabel('Longitude')
# ax2.set_ylabel('Latitude')
# scatter = ax2.scatter(lon_land,lat_land,c=frequencies_df[2],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax2,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax2.coastlines()
# ax2.title.set_text('Decision 2: % samples optimal')

# # Percent of time that decision 3 is optimal
# ax3.set_xlabel('Longitude')
# ax3.set_ylabel('Latitude')
# scatter = ax3.scatter(lon_land,lat_land,c=frequencies_df[3],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax3,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax3.coastlines()
# ax3.title.set_text('Decision 3: % samples optimal')

# plt.show()

###########################################################################
# FIGURE 3B
# Varying risk AND decision inputs
# Plot number of optimal decisions per cell
n_samples = 200

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = './data/decision_files_' + str(n_samples) + '/'
file_list = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
n_files = len(file_list)

nloc_land = 1711
Y_riskdec = np.empty(n_files*nloc_land).reshape(n_files,nloc_land)

# Loop over files in the folder
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    Y_riskdec[i, :] = data['optimal_decision']

noptions_riskdec = np.empty(nloc_land)
for i in range(1711):
    noptions_riskdec[i] = len(np.unique(Y_riskdec[:,i]))
np.unique(noptions_riskdec, return_counts = True)

# Plot number of decisions optimal in each cell
fig = plt.figure(figsize=(18,15))
lon_land = data['lon']
lat_land = data['lat']
ax1 = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
scatter = ax1.scatter(lon_land,lat_land,c=noptions_riskdec,s=12,cmap=dis_seq, norm=dis_seq_norm)
ax1.legend(handles=legend_elements)
ax1.coastlines()
plt.show()

# # Plot percentage of time that each decision was optimal in a cell


# # List of all possible decisions
# decision_options = np.unique(Y_riskdec, return_counts = True)

# # Calculate frequencies of each decision per location
# column_frequencies = {value: [] for value in decision_options[0]}
# num_rows = Y.shape[0]

# for col_index in range(Y.shape[1]):
#     unique_values, value_counts = np.unique(Y[:, col_index], return_counts=True)
#     # Create a dictionary to hold frequency counts for current column
#     freq_dict = dict(zip(unique_values, value_counts))
#     # Calculate frequencies for each decision option
#     for value in decision_options[0]:
#         if value in freq_dict:
#             frequency = freq_dict[value] / num_rows
#         else:
#             frequency = 0.0  # Value not present, set frequency to 0
        
#         column_frequencies[value].append(frequency)

# frequencies_df = pd.DataFrame(column_frequencies)

# fig = plt.figure(figsize=(18,9))
# ax1 = plt.subplot(1,3,1,projection=ccrs.PlateCarree())
# ax2 = plt.subplot(1,3,2,projection=ccrs.PlateCarree())
# ax3 = plt.subplot(1,3,3,projection=ccrs.PlateCarree())
# lon_land = data['lon']
# lat_land = data['lat']

# # Percent of time that decision 1 is optimal
# ax1.set_xlabel('Longitude')
# ax1.set_ylabel('Latitude')
# scatter = ax1.scatter(lon_land,lat_land,c=frequencies_df[1],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax1,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax1.coastlines()
# ax1.title.set_text('Decision 1: % samples optimal')

# # Percent of time that decision 2 is optimal
# ax2.set_xlabel('Longitude')
# ax2.set_ylabel('Latitude')
# scatter = ax2.scatter(lon_land,lat_land,c=frequencies_df[2],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax2,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax2.coastlines()
# ax2.title.set_text('Decision 2: % samples optimal')

# # Percent of time that decision 3 is optimal
# ax3.set_xlabel('Longitude')
# ax3.set_ylabel('Latitude')
# scatter = ax3.scatter(lon_land,lat_land,c=frequencies_df[3],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax3,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax3.coastlines()
# ax3.title.set_text('Decision 3: % samples optimal')

# plt.show()

###########################################################################
# FIGURE 3C
# Bar graph comparing number of optimal decisions
# Varying only risk- vs. varying risk- and decision-related attributes

# List of all possible decisions
decision_options = np.unique(Y_risk, return_counts = True)

counts_risk = np.array([np.sum(noptions_risk == num) for num in decision_options[0]])
counts_riskdec = np.array([np.sum(noptions_riskdec == num) for num in decision_options[0]])

bar_width = 0.4

fig, ax = plt.subplots()
ax.bar(decision_options[0]-bar_width/2, counts_risk, width=bar_width,
       color = dis_non_cols[0],
       label='Varying risk only')
ax.bar(decision_options[0]+bar_width/2, counts_riskdec, width=bar_width,
       color = dis_non_cols[1],
       label='Varying risk and decision inputs')
ax.set_xlabel('Number of optimal decisions')
ax.set_ylabel('Number of cells')
ax.set_xticks(decision_options[0])
ax.legend()
plt.show()

###########################################################################
# FIGURE 4
# Boxplots of sensitivity metric (KS statistic for risk, PMF MVD for decision)
# For three chosen locations
lon_ind = 241 # London
ld_ind = 1000 # Lake District
scot_ind = 1445 # location in Scotland very sensitive to SSP

# Read in sensitivity results:
# For risk
KS_vals = np.load('./data/pawn_results/KS_vals_lhc200.npy')
KS_lbs = np.load('./data/pawn_results/KS_lbs_lhc200.npy')
KS_ubs = np.load('./data/pawn_results/KS_ubs_lhc200.npy')
# For decision
max_dist_vals = np.load('./data/pawn_results/max_dist_vals_lhc200.npy')
max_dist_lbs = np.load('./data/pawn_results/max_dist_lbs_lhc200.npy')
max_dist_ubs = np.load('./data/pawn_results/max_dist_ubs_lhc200.npy')

plt.figure(figsize=(19,16))
# Risk-related boxplots:
# For London
ax1 = plt.subplot(3,2,1)
pf.boxplot1(KS_vals[:,lon_ind],
            S_lb=KS_lbs[:,lon_ind],
            S_ub=KS_ubs[:,lon_ind],
            Y_Label="Mean KS Statistic"
            )
plt.title('(a)')
ax1.set_xticklabels(['']*len(ax1.get_xticks()))
plt.ylim((None, 0.5))
# For Lake District
ax2 = plt.subplot(3,2,3)
pf.boxplot1(KS_vals[:,ld_ind],
            S_lb=KS_lbs[:,ld_ind],
            S_ub=KS_ubs[:,ld_ind],
            Y_Label="Mean KS Statistic"
            )
plt.title('(c)')
ax2.set_xticklabels(['']*len(ax2.get_xticks()))
plt.ylim((None, 0.5))
# For Scotland
plt.subplot(3,2,5)
pf.boxplot1(KS_vals[:,scot_ind],
            S_lb=KS_lbs[:,scot_ind],
            S_ub=KS_ubs[:,scot_ind],
            Y_Label="Mean KS Statistic",
            X_Labels=X_labels_risk_short)
plt.title('(e)')
plt.xticks(fontsize=11)
# plt.xticks(rotation=30,ha='right')
plt.ylim((None, 0.5))

# Decision-related boxplots:
# For London
ax4 = plt.subplot(3,2,2)
pf.boxplot1(max_dist_vals[:,lon_ind],
            S_lb=max_dist_lbs[:,lon_ind],
            S_ub=max_dist_ubs[:,lon_ind],
            Y_Label="Maximum MVD"
            )
plt.title('(b)')
ax4.set_xticklabels(['']*len(ax4.get_xticks()))
plt.ylim((None, 0.5))
# For Lake District
ax5 = plt.subplot(3,2,4)
pf.boxplot1(max_dist_vals[:,ld_ind],
            S_lb=max_dist_lbs[:,ld_ind],
            S_ub=max_dist_ubs[:,ld_ind],
            Y_Label="Maximum MVD"
            )
plt.title('(d)')
ax5.set_xticklabels(['']*len(ax5.get_xticks()))
plt.ylim((None, 0.5))
# For Scotland
plt.subplot(3,2,6)
pf.boxplot1(max_dist_vals[:,scot_ind],
            S_lb=max_dist_lbs[:,scot_ind],
            S_ub=max_dist_ubs[:,scot_ind],
            Y_Label="Maximum MVD",
            X_Labels=X_labels_short)
plt.title('(f)')
plt.xticks(fontsize=11)
# plt.xticks(rotation=30,ha='right')
plt.ylim((None, 0.5))

plt.show()

###########################################################################
# FIGURE 7
# Map of decision sensitivity values for all inputs

# can be any data file
data = pd.read_csv("./data/decision_files_jit/OptimalDecision_ssp1_2deg_ChangeFactor_v1_53.78_v2_-3.804_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv")
lon = data['lon']
lat = data['lat']

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
cp = ax1.scatter(lon,lat,c=max_dist_vals[0,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax1,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[0])
ax1.coastlines(linewidth=0.5)
ax1.title.set_text('(a)')

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(lon,lat,c=max_dist_vals[1,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax2,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[1])
ax2.coastlines(linewidth=0.5)
ax2.title.set_text('(b)')

ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
cp = ax3.scatter(lon,lat,c=max_dist_vals[2,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax3,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[2])
ax3.coastlines(linewidth=0.5)
ax3.title.set_text('(c)')

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
cp = ax4.scatter(lon,lat,c=max_dist_vals[3,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax4,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[3])
ax4.coastlines(linewidth=0.5)
ax4.title.set_text('(d)')

ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
cp = ax5.scatter(lon,lat,c=max_dist_vals[4,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax5,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[4])
ax5.coastlines(linewidth=0.5)
ax5.title.set_text('(e)')

ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
cp = ax6.scatter(lon,lat,c=max_dist_vals[5,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax6,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[5])
ax6.coastlines(linewidth=0.5)
ax6.title.set_text('(f)')

ax7.set_xlabel('Longitude')
ax7.set_ylabel('Latitude')
cp = ax7.scatter(lon,lat,c=max_dist_vals[6,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax7,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[6])
ax7.coastlines(linewidth=0.5)
ax7.title.set_text('(g)')

ax8.set_xlabel('Longitude')
ax8.set_ylabel('Latitude')
cp = ax8.scatter(lon,lat,c=max_dist_vals[7,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax8,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[7])
ax8.coastlines(linewidth=0.5)
ax8.title.set_text('(h)')

ax9.set_xlabel('Longitude')
ax9.set_ylabel('Latitude')
cp = ax9.scatter(lon,lat,c=max_dist_vals[8,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax9,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[8])
ax9.coastlines(linewidth=0.5)
ax9.title.set_text('(i)')

ax10.set_xlabel('Longitude')
ax10.set_ylabel('Latitude')
cp = ax10.scatter(lon,lat,c=max_dist_vals[9,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax10,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[9])
ax10.coastlines(linewidth=0.5)
ax10.title.set_text('(j)')

ax11.set_xlabel('Longitude')
ax11.set_ylabel('Latitude')
cp = ax11.scatter(lon,lat,c=max_dist_vals[10,:],vmin=0,vmax=0.5,s=5,cmap='Blues')
cbar = plt.colorbar(cp,ax=ax11,shrink=0.7)
cbar.set_label('Mean maximum distance for\n' + X_labels[10])
ax11.coastlines(linewidth=0.5)
ax11.title.set_text('(k)')

plt.show()