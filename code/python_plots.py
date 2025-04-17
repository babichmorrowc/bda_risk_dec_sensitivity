import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import safepython.plot_functions as pf 
from netCDF4 import Dataset

os.chdir("./code")
from bda_functions import *

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
                  'DAYCOST',
                  'COST-M',
                  'COST-C',
                  'EFF-M',
                  'EFF-C',
                  'WEIGHT']
X_labels_risk_short = X_labels_short[0:5]

# Chosen locations
lon_ind = 241 # London
ld_ind = 1000 # Lake District
scot_ind = 1445 # location in Scotland very sensitive to SSP
chosen_ind = [lon_ind, ld_ind, scot_ind]

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
# FIGURE 2A & 2B
# Mean and standard deviation of risk in each grid cell
# Roughly following Laura's code here:
# https://github.com/babichmorrowc/compass_miniproject/blob/main/code/orig_laura_code/plots_for_paper_python.py#L471
X_risk = np.array(np.meshgrid(ssp_opts, warming_opts, calibration_opts, vuln1_opts, vuln2_opts)).T.reshape(-1,5)

# can't make full output matrix - memory issues, so calculate combined SD following http://www.obg.cuhk.edu.hk/ResearchSupport/StatTools/CombineMeansSDs_Pgm.php#:~:text=The%20Standard%20Error%20of%20the,sizes%20from%20all%20the%20groups.
txx = np.empty(110*83*X_risk.shape[0]).reshape(110,83,X_risk.shape[0])
tx = np.empty(110*83*X_risk.shape[0]).reshape(110,83,X_risk.shape[0])

# Pretty slow -- can I jit this somehow?
for i in range(X_risk.shape[0]):
    data = Dataset('../data/'+X_risk[i,2]+
                    '/GAMsamples_expected_annual_impact_data_'+X_risk[i,2]+
                    '_WL'+X_risk[i,1]+
                    '_SSP'+X_risk[i,0]+
                    '_vp1='+X_risk[i,3]+'_vp2='+X_risk[i,4]+'.nc')
    allEAI = np.array(data.variables['sim_annual_impact'])
    # land locations have EAI < 9e30
    allEAI[np.where(allEAI > 9e30)] = 0
    allEAI = 10**allEAI - 1 # 110 x 83 x 1000
    # get sum of squared eai values across all gam samples for each combo of risk parameters
    txx[:,:,i] = np.apply_along_axis(lambda x : sum(x**2), 2, allEAI) # 110 x 83 x 162
    # get sum of eai values across all gam samples
    tx[:, :, i] = np.apply_along_axis(lambda x: sum(x), 2, allEAI) # 110 x 83 x 162

# Sum txx and tx across all risk parameter combinations
txx_all1 = np.apply_along_axis(lambda x : sum(x), 2, txx) # 110 x 83
tx_all1 = np.apply_along_axis(lambda x : sum(x), 2, tx) # 110 x 83

SDall = np.empty(110*83).reshape(110,83)
for i in range(110):
    for j in range(83):
        SDall[i,j] = np.sqrt((txx_all1[i,j] - tx_all1[i,j]**2/(X_risk.shape[0]*1000))/(X_risk.shape[0]*1000 -1))
nloc = 110*83
SDall = SDall.reshape(nloc)

# Now averaging over all values from all risk parameter combinations
meanall = tx_all1 / (X_risk.shape[0]*1000)
meanall[meanall == 0] = np.nan
log_norm_mean2 = LogNorm(vmin = 4e-2, vmax = 9e5)

lon = np.array(data.variables['exposure_longitude'])
lon = lon.reshape(nloc)
lat = np.array(data.variables['exposure_latitude'])
lat = lat.reshape(nloc)
log_norm_sd = LogNorm(vmin=1.0e1, vmax=2.0e6)

# Get the risk values in the three chosen locations
# Get exposure values at the default risk parameters
def_exp = get_Exp("../data/", ssp="2", ssp_year=2041) # 110 x 83
# Filter to land locations
land_ind = np.where(def_exp < 9e30)
def_exp[land_ind][chosen_ind]
meanall[land_ind][chosen_ind]

###########################################################################
# FIGURE 2C
# Varying risk inputs ONLY
# Plot number of optimal decisions per cell

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = '../data/decision_files_jit/'
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

# Get the number of optimal decisions in the 3 chosen locations
noptions_risk[chosen_ind]

###########################################################################
# FIGURE 2D
# Varying risk AND decision inputs
# Plot number of optimal decisions per cell
n_samples = 200

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = '../data/decision_files_' + str(n_samples) + '/'
file_list = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
n_files = len(file_list)

Y_riskdec = np.empty(n_files*nloc_land).reshape(n_files,nloc_land)

# Loop over files in the folder
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    Y_riskdec[i, :] = data['optimal_decision']

noptions_riskdec = np.empty(nloc_land)
for i in range(nloc_land):
    noptions_riskdec[i] = len(np.unique(Y_riskdec[:,i]))
# np.unique(noptions_riskdec, return_counts = True)

###########################################################################
# FIGURE 2 PIECHARTS
# Inset piecharts showing number of optimal decisions
# Varying only risk- vs. varying risk- and decision-related attributes

# List of all possible decisions
decision_options = np.unique(Y_risk, return_counts = True)

counts_risk = np.array([np.sum(noptions_risk == num) for num in decision_options[0]])
counts_riskdec = np.array([np.sum(noptions_riskdec == num) for num in decision_options[0]])

################################################################################################
# FIGURE 2

lon_land = data['lon']
lat_land = data['lat']

# Make 4 panel figure
fig, axs = plt.subplots(2, 2,
                        figsize=(8.9,10.8),
                        layout="constrained",
                        subplot_kw={'projection': ccrs.PlateCarree()})
# Plot of mean risk
cp = axs[0,0].scatter(lon,lat,c=meanall,
                    norm=log_norm_mean2,
                    s=11,
                    cmap='Purples')
highlight0 = axs[0,0].scatter(lon_land[chosen_ind],
                           lat_land[chosen_ind],
                           facecolors = 'none',
                           edgecolors = 'black',
                           s=30)
# Create an inset axis for the colorbar (horizontal, top left)
# cax = inset_axes(axs[0,0], width="45%", height="5%", loc='upper left', borderpad=1.5)
# cbar = plt.colorbar(cp, ax=axs[0,0], cax=cax, orientation='horizontal')
# cbar.set_label('Mean EAI', labelpad=4)
# cbar.ax.xaxis.set_label_position('top') 
# cax.text(0.5, -1.2, '(Annual person-days lost)', transform=cax.transAxes,
#          ha='center', va='top', fontsize=8)
cax = inset_axes(axs[0,0], width="5%", height="50%", loc='upper right', borderpad=4)
cbar = plt.colorbar(cp, ax=axs[0,0], cax=cax, orientation='vertical')
cbar.set_label('Mean EAI', labelpad=10)
cbar.ax.yaxis.set_label_position('left')
cax.text(3, 0.5, '(Annual person-days lost)', transform=cax.transAxes,
         ha='left', va='center', rotation=90, fontsize=9)
axs[0,0].set_title('(a)')
axs[0,0].coastlines()

# Plot of standard deviation risk
cp = axs[0,1].scatter(lon,lat,c=SDall,
                 norm=log_norm_sd,
                 s=11,
                 cmap='Purples')
highlight1 = axs[0,1].scatter(lon_land[chosen_ind],
                           lat_land[chosen_ind],
                           facecolors = 'none',
                           edgecolors = 'black',
                           s=30)
# Create an inset axis for the colorbar
cax = inset_axes(axs[0,1], width="5%", height="50%", loc='upper right', borderpad=4)
cbar = plt.colorbar(cp, ax=axs[0,1], cax=cax, orientation='vertical')
cbar.set_label('Standard deviation of EAI', labelpad=10)
cbar.ax.yaxis.set_label_position('left')
cax.text(3, 0.5, '(Annual person-days lost)', transform=cax.transAxes,
         ha='left', va='center', rotation=90, fontsize=9)
axs[0,1].set_title('(b)')
axs[0,1].coastlines()

# Plot number of decisions optimal in each cell
# Varying only risk
scatter = axs[1,0].scatter(lon_land,lat_land,
                           c=noptions_risk,s=12,cmap=dis_seq, norm=dis_seq_norm)
highlight2 = axs[1,0].scatter(lon_land[chosen_ind],
                           lat_land[chosen_ind],
                           facecolors = 'none',
                           edgecolors = 'black',
                           s=30)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=legend_labs[i], 
           markersize=10, markerfacecolor=dis_seq_cols[i])
    for i in range(len(legend_labs))
]
axs[1,0].legend(handles=legend_elements)
axs[1,0].set_title('(c)')
axs[1,0].coastlines()
# Add a subplot for panel b
pie1 = inset_axes(axs[1,0], width="30%", height="30%", loc='upper right', borderpad = 1)
pie1.pie(counts_risk, colors = dis_seq_cols)

# Plot number of decisions optimal in each cell
# Varying risk and decision parameters
axs[1,1].set_xlabel('Longitude')
axs[1,1].set_ylabel('Latitude')
scatter = axs[1,1].scatter(lon_land,lat_land,c=noptions_riskdec,s=12,cmap=dis_seq, norm=dis_seq_norm)
highlight3 = axs[1,1].scatter(lon_land[chosen_ind],
                           lat_land[chosen_ind],
                           facecolors = 'none',
                           edgecolors = 'black',
                           s=30)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=legend_labs[i], 
           markersize=10, markerfacecolor=dis_seq_cols[i])
    for i in range(len(legend_labs))
]
axs[1,1].legend(handles=legend_elements)
axs[1,1].set_title('(d)')
axs[1,1].coastlines()

# Add a subplot for panel c
pie2 = inset_axes(axs[1,1], width="30%", height="30%", loc='upper right', borderpad = 1)
pie2.pie(counts_riskdec, colors = dis_seq_cols)

# Make sure plot (a) and (b) have the same limits as (c) and (d)
axs[0,0].set_xlim(axs[1,0].get_xlim())
axs[0,0].set_ylim(axs[1,0].get_ylim())
axs[0,1].set_xlim(axs[1,0].get_xlim())
axs[0,1].set_ylim(axs[1,0].get_ylim())

# plt.subplots_adjust(wspace=0.1)
plt.savefig('../figures/risk-dec-uncertainty.png')
plt.show()

##############################################################################
# FIGURE 3
# Maps of the proportion of the samples where each decision is optimal in each cell

# Varying only risk
# Calculate frequencies of each decision per location
column_frequencies_risk = {value: [] for value in decision_options[0]}
num_rows_risk = Y_risk.shape[0]

for col_index in range(Y_risk.shape[1]):
    unique_values, value_counts = np.unique(Y_risk[:, col_index], return_counts=True)
    # Create a dictionary to hold frequency counts for current column
    freq_dict = dict(zip(unique_values, value_counts))
    # Calculate frequencies for each decision option
    for value in decision_options[0]:
        if value in freq_dict:
            frequency = freq_dict[value] / num_rows_risk
        else:
            frequency = 0.0  # Value not present, set frequency to 0
        
        column_frequencies_risk[value].append(frequency)
frequencies_df_risk = pd.DataFrame(column_frequencies_risk)

# Varying risk and decision-related parameters
# Calculate frequencies of each decision per location
column_frequencies_riskdec = {value: [] for value in decision_options[0]}
num_rows_riskdec = Y_riskdec.shape[0]

for col_index in range(Y_riskdec.shape[1]):
    unique_values, value_counts = np.unique(Y_riskdec[:, col_index], return_counts=True)
    # Create a dictionary to hold frequency counts for current column
    freq_dict = dict(zip(unique_values, value_counts))
    # Calculate frequencies for each decision option
    for value in decision_options[0]:
        if value in freq_dict:
            frequency = freq_dict[value] / num_rows_riskdec
        else:
            frequency = 0.0  # Value not present, set frequency to 0
        
        column_frequencies_riskdec[value].append(frequency)
frequencies_df_riskdec = pd.DataFrame(column_frequencies_riskdec)

fig = plt.figure(figsize=(18,9))
ax1 = plt.subplot(2,3,1,projection=ccrs.PlateCarree())
ax2 = plt.subplot(2,3,2,projection=ccrs.PlateCarree())
ax3 = plt.subplot(2,3,3,projection=ccrs.PlateCarree())
ax4 = plt.subplot(2,3,4,projection=ccrs.PlateCarree())
ax5 = plt.subplot(2,3,5,projection=ccrs.PlateCarree())
ax6 = plt.subplot(2,3,6,projection=ccrs.PlateCarree())
lon_land = data['lon']
lat_land = data['lat']

# Percent of time that decision 1 is optimal
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
scatter = ax1.scatter(lon_land,lat_land,c=frequencies_df_risk[1],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax1, shrink=0.8)
cbar.set_label('Do nothing: % samples optimal')
ax1.coastlines()
ax1.title.set_text('(a)')

# Percent of time that decision 2 is optimal
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
scatter = ax2.scatter(lon_land,lat_land,c=frequencies_df_risk[2],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax2,shrink=0.8)
cbar.set_label('Modify hours: % samples optimal')
ax2.coastlines()
ax2.title.set_text('(b)')

# Percent of time that decision 3 is optimal
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
scatter = ax3.scatter(lon_land,lat_land,c=frequencies_df_risk[3],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax3,shrink=0.8)
cbar.set_label('Buy cooling equipment: % samples optimal')
ax3.coastlines()
ax3.title.set_text('(c)')

# Percent of time that decision 1 is optimal
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
scatter = ax4.scatter(lon_land,lat_land,c=frequencies_df_riskdec[1],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax4,shrink=0.8)
cbar.set_label('Do nothing: % samples optimal')
ax4.coastlines()
ax4.title.set_text('(d)')

# Percent of time that decision 2 is optimal
ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
scatter = ax5.scatter(lon_land,lat_land,c=frequencies_df_riskdec[2],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax5,shrink=0.8)
cbar.set_label('Modify hours: % samples optimal')
ax5.coastlines()
ax5.title.set_text('(e)')

# Percent of time that decision 3 is optimal
ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
scatter = ax6.scatter(lon_land,lat_land,c=frequencies_df_riskdec[3],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax6,shrink=0.8)
cbar.set_label('Buy cooling equipment: % samples optimal')
ax6.coastlines()
ax6.title.set_text('(f)')

plt.savefig('../figures/prop-dec-maps.png')
plt.show()

###########################################################################
# FIGURE 3
# Boxplots of sensitivity metric (KS statistic for risk, PMF MVD for decision)
# For three chosen locations

# Read in sensitivity results:
# For risk
KS_vals = np.load('../data/pawn_results/KS_vals_lhc200.npy')
KS_lbs = np.load('../data/pawn_results/KS_lbs_lhc200.npy')
KS_ubs = np.load('../data/pawn_results/KS_ubs_lhc200.npy')
# For decision
max_dist_vals = np.load('../data/pawn_results/max_dist_vals_lhc200.npy')
max_dist_lbs = np.load('../data/pawn_results/max_dist_lbs_lhc200.npy')
max_dist_ubs = np.load('../data/pawn_results/max_dist_ubs_lhc200.npy')

# Barplot error bar calculations
KS_barlower = KS_vals - KS_lbs
KS_barupper = KS_ubs - KS_vals
max_dist_barlower = max_dist_vals - max_dist_lbs
max_dist_barupper = max_dist_ubs - max_dist_vals

# Barplot with error bars
barwidth = 0.2
fig = plt.figure(figsize=(15,10))
# London
ax1 = plt.subplot(3,1,1)
ax1.bar(x=np.arange(len(X_labels_risk)) - barwidth/2,
        width=barwidth,
        height=KS_vals[:,lon_ind],
        yerr=np.array([KS_barlower[:,lon_ind], KS_barupper[:,lon_ind]]))
ax1.bar(x=np.arange(len(X_labels)) + barwidth/2,
        width=barwidth,
        height=max_dist_vals[:,lon_ind],
        yerr=np.array([max_dist_barlower[:,lon_ind], max_dist_barupper[:,lon_ind]]))
ax1.set_title("(a)")
ax1.set_ylabel("Sensitivity")
ax1.legend(labels = ["Sensitivity of risk", "Sensitivity of decision"])
ax1.set_xticks(np.arange(len(X_labels)))
ax1.set_xticklabels(X_labels_short)
ax1.set_ylim(0,0.45)
# Lake District
ax2 = plt.subplot(3,1,2)
ax2.bar(x=np.arange(len(X_labels_risk)) - barwidth/2,
        width=barwidth,
        height=KS_vals[:,ld_ind],
        yerr=np.array([KS_barlower[:,ld_ind], KS_barupper[:,ld_ind]]))
ax2.bar(x=np.arange(len(X_labels)) + barwidth/2,
        width=barwidth,
        height=max_dist_vals[:,ld_ind],
        yerr=np.array([max_dist_barlower[:,ld_ind], max_dist_barupper[:,ld_ind]]))
ax2.set_title("(b)")
ax2.set_ylabel("Sensitivity")
ax2.legend(labels = ["Sensitivity of risk", "Sensitivity of decision"])
ax2.set_xticks(np.arange(len(X_labels)))
ax2.set_xticklabels(X_labels_short)
ax2.set_ylim(0,0.45)
# Scotland
ax3 = plt.subplot(3,1,3)
ax3.bar(x=np.arange(len(X_labels_risk)) - barwidth/2,
        width=barwidth,
        height=KS_vals[:,scot_ind],
        yerr=np.array([KS_barlower[:,scot_ind], KS_barupper[:,scot_ind]]))
ax3.bar(x=np.arange(len(X_labels)) + barwidth/2,
        width=barwidth,
        height=max_dist_vals[:,scot_ind],
        yerr=np.array([max_dist_barlower[:,scot_ind], max_dist_barupper[:,scot_ind]]))
ax3.set_title("(c)")
ax3.set_ylabel("Sensitivity")
ax3.legend(labels = ["Sensitivity of risk", "Sensitivity of decision"])
ax3.set_xticks(np.arange(len(X_labels)))
ax3.set_xticklabels(X_labels_short)
ax3.set_ylim(0,0.45)

plt.savefig('../figures/sens-3locs.png')
plt.show()

###########################################################################
# # FIGURE 4
# # Stacked barplot of sensitivity in each cell by latitude

# # can be any data file
# data = pd.read_csv("../data/decision_files_jit/OptimalDecision_ssp1_2deg_ChangeFactor_v1_53.78_v2_-3.804_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv")

# # Create a dataframe with the latitude data and the sensitivity values
# max_dist_vals_t = pd.DataFrame(np.transpose(max_dist_vals)) # 1711 x 11
# max_dist_vals_t.columns = X_labels_short
# # Join on and sort by latitude
# lat_cell_data = data.join(max_dist_vals_t).sort_values(by = 'lat')
# lat_cell_data = lat_cell_data[['lat'] + X_labels_short].set_index('lat')

# # # Create a stacked barplot
# # ax = lat_cell_data.plot(kind='bar', stacked=True, figsize=(15, 10))
# # plt.xlabel('')
# # plt.ylabel('Maximum MVD')
# # plt.legend(title='Input parameters', bbox_to_anchor=(1.05, 1), loc='upper left')

# # # Create a secondary x-axis
# # secax = ax.secondary_xaxis('bottom')
# # secax.set_xlabel('Latitude')
# # # Set x-ticks to label every 1 degree of latitude
# # min_lat = lat_cell_data.index.min()
# # max_lat = lat_cell_data.index.max()
# # lat_ticks = np.arange(min_lat, max_lat + 1, 1)
# # # Map latitude values to the positions of the bars
# # bar_positions = np.linspace(0, len(lat_cell_data) - 1, len(lat_cell_data))
# # lat_positions = np.interp(lat_ticks, lat_cell_data.index, bar_positions)
# # secax.set_xticks(lat_positions)
# # secax.set_xticklabels([f'{tick:.1f}' for tick in lat_ticks])
# # # Hide the original x-tick labels
# # ax.set_xticks([])

# # # Show the plot
# # plt.tight_layout()
# # plt.show()

# # Consolidating inputs into categories
# # Sum of all values in the category
# # Maximum of the values in the category
# # Proportion of total
# lat_cell_data['Risk'] = lat_cell_data[X_labels_risk_short].sum(axis=1)
# lat_cell_data['Risk1'] = np.amax(lat_cell_data[X_labels_risk_short], axis=1)
# lat_cell_data['Cost per day'] = lat_cell_data['COST1'] # both a max and a sum
# lat_cell_data['Decision costs'] = lat_cell_data[['COST2', 'COST3']].sum(axis=1)
# lat_cell_data['Decision costs1'] = np.amax(lat_cell_data[['COST2', 'COST3']], axis=1)
# lat_cell_data['Decision efficacies'] = lat_cell_data[['EFF1', 'EFF2']].sum(axis=1)
# lat_cell_data['Decision efficacies1'] = np.amax(lat_cell_data[['EFF1', 'EFF2']], axis=1)
# lat_cell_data['Weight'] = lat_cell_data['WEIGHT'] # both a max and a sum

# lat_cell_data_cons = lat_cell_data[['Risk', 'Cost per day', 'Decision costs', 'Decision efficacies', 'Weight']]
# lat_cell_data_maxes = lat_cell_data[['Risk1', 'Cost per day', 'Decision costs1', 'Decision efficacies1', 'Weight']]
# lat_cell_data_props = lat_cell_data_cons.div(lat_cell_data_cons.sum(axis=1), axis=0)
# lat_cell_data_maxprops = lat_cell_data_maxes.div(lat_cell_data_maxes.sum(axis=1), axis=0)

# # Create a stacked barplot
# ax = lat_cell_data_cons.plot(kind='bar', stacked=True, figsize=(15, 10))
# plt.xlabel('')
# plt.ylabel('Total Maximum MVD')
# plt.legend(title='Input parameter type', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Create a secondary x-axis
# secax = ax.secondary_xaxis('bottom')
# secax.set_xlabel('Latitude')
# # Set x-ticks to label every 1 degree of latitude
# min_lat = lat_cell_data.index.min()
# max_lat = lat_cell_data.index.max()
# lat_ticks = np.arange(min_lat, max_lat + 1, 1)
# # Map latitude values to the positions of the bars
# bar_positions = np.linspace(0, len(lat_cell_data) - 1, len(lat_cell_data))
# lat_positions = np.interp(lat_ticks, lat_cell_data.index, bar_positions)
# secax.set_xticks(lat_positions)
# secax.set_xticklabels([f'{tick:.1f}' for tick in lat_ticks])
# # Hide the original x-tick labels
# ax.set_xticks([])

# # Show the plot
# plt.tight_layout()
# plt.show()

# # Create rotated barplot with the map
# # Create a figure with GridSpec to align the plots
# fig = plt.figure(figsize=(20, 10))
# gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

# # Create the horizontal bar plot
# ax1 = fig.add_subplot(gs[0, 0])
# lat_cell_data_props.plot(kind='barh', stacked=True, ax=ax1)
# ax1.set_xlabel('Proportion of Total')
# ax1.set_ylabel('Latitude')
# ax1.legend(title='Input parameter type', bbox_to_anchor=(1.05, 1), loc='upper left')
# # Map latitude values to bar positions
# lat_values = lat_cell_data.index.to_numpy()  # Extract latitudes from DataFrame index
# bar_positions = np.arange(len(lat_values))  # Get categorical positions of bars

# # Generate tick positions at real latitude values
# lat_ticks = np.arange(lat_values.min(), lat_values.max() + 1, 1)
# lat_tick_positions = np.interp(lat_ticks, lat_values, bar_positions)  # Reverse order

# # Set the y-ticks and labels
# ax1.set_ylim(lat_values.min(), lat_values.max())
# ax1.set_yticks(lat_tick_positions)
# ax1.set_yticklabels([f'{tick:.1f}' for tick in lat_ticks])

# # Create the map of the UK
# ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
# ax2.set_extent([-10, 2, lat_values.min(), lat_values.max()], crs=ccrs.PlateCarree())  # Set the extent to focus on the UK
# ax2.coastlines()
# # Manually set the same y-ticks on the map
# ax2.set_yticks(lat_ticks)
# ax2.set_yticklabels([f'{tick:.1f}' for tick in lat_ticks])

# # Adjust layout
# plt.tight_layout()
# plt.show()

###########################################################################
# FIGURE 5
# Map of decision sensitivity values for all inputs

# can be any data file
data = pd.read_csv("./data/decision_files_jit/OptimalDecision_ssp1_2deg_ChangeFactor_v1_53.78_v2_-3.804_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv")
lon = data['lon']
lat = data['lat']

fig, axes = plt.subplots(5, 5, figsize=(10, 13), subplot_kw={'projection': ccrs.PlateCarree()})

# Define subplot positions to match your custom layout
plot_indices = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),  # First row
    (1, 0),                                 # Second row (one plot)
    (2, 0), (2, 1),                          # Third row (two plots)
    (3, 0), (3, 1),
    (4, 0)                   
]

# Iterate through each axis and plot
param_cats = ['Risk', 'Cost per day', 'Decision costs', 'Decision efficacies', 'Weight']
for i, (row, col) in enumerate(plot_indices):
    ax = axes[row, col]
    cp = ax.scatter(lon, lat, c=max_dist_vals[i, :], vmin=0, vmax=0.5, s=5, cmap='Blues')
    ax.coastlines(linewidth=0.5)
    ax.set_title(f'({chr(97 + i)})\n\n{X_labels_short[i]}')  # (a), (b), (c), etc.
    if col == 0:
        ax.text(-0.07, 0.55, param_cats[row], va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=ax.transAxes, size='x-large')

# Insert one large color bar
# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=0.5))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, location='right', shrink = 0.6)
cbar.set_label('Decision Sensitivity')

# Hide empty subplots
for row in range(5):
    for col in range(5):
        if (row, col) not in plot_indices:
            fig.delaxes(axes[row, col])

plt.tight_layout()
plt.savefig('../figures/dec-sens-maps.png')
plt.show()

#############################################################################
# SUPPLEMENT
#############################################################################
# Plot loss functions and risk for 3 locations

# Set up default decision attributes
# Cost per day of work lost:
cost_per_day = 200
# Cost attributes of each of the 3 decisions + objective scores
def_dec_attributes = np.array([[0, 0, 5],
                           [250, 0.4, 6],
                           [600, 0.8, 4]])
# Relative weighting of priorities
c_weight_cost = 0.8 
cweights = [float(c_weight_cost),1-float(c_weight_cost)]

# Get exposure values at the default risk parameters
def_exp = get_Exp("../data/", ssp="2", ssp_year=2041) # 110 x 83
# Filter to land locations
land_ind = np.where(def_exp < 9e30)
def_exp = def_exp[land_ind]
# Get risk values at the default risk parameters
def_eai = get_EAI(input_data_path="../data/", data_source="UKCP_BC", warming_level="2deg", ssp="2", vp1="54.5", vp2="-4.1")
# Filter to land locations
def_eai = def_eai[land_ind]

# Define a sequence of numbers for x axis
x = np.linspace(0,6,200)

# Define inputs
nd = 3

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, ax in zip(chosen_ind, axes):
    p = def_exp[idx]
    if p == 0: # setting exposure in Scotland to 1
        p = 1
    cost = np.empty([nd, len(x)])
    for j in range(nd):
        for k in range(len(x)):
            cost[j, k] = def_dec_attributes[j, 0] * p + cost_per_day * (1 - def_dec_attributes[j, 1]) * (10 ** x[k])
    log_cost = np.log10(cost)
    ax.plot(x, log_cost[0, :], label='Do nothing', color='black')
    ax.plot(x, log_cost[1, :], label='Modify working hours', color='lawngreen')
    ax.plot(x, log_cost[2, :], label='Buy cooling equipment', color='magenta')
    ax.set_xlabel('EAI in grid cell (log base 10)')
    ax.set_ylabel('Cost, £ (log base 10)')
    ax.legend()
    ax.hist(def_eai[idx], density='False', color='grey', rwidth=0.8)

axes[0].title.set_text('(a)')
axes[1].title.set_text('(b)')
axes[2].title.set_text('(c)')

plt.tight_layout()
plt.savefig('../figures/supplement_loss_functions.png')
plt.show()
