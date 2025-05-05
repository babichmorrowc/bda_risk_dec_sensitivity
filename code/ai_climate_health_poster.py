import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import safepython.plot_functions as pf 
from netCDF4 import Dataset

os.chdir("./code")
from bda_functions import *
from plotting_functions import *

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
lon_ind = 266 # London
ld_ind = 1058 # Lake District
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

for i in range(X_risk.shape[0]): # 162
    # print(i)
    data = Dataset('../data/'+X_risk[i,2]+
                    '/GAMsamples_expected_annual_impact_data_'+X_risk[i,2]+
                    '_WL'+X_risk[i,1]+
                    '_SSP'+X_risk[i,0]+
                    '_vp1='+X_risk[i,3]+'_vp2='+X_risk[i,4]+'.nc')
    allEAI = np.array(data.variables['sim_annual_impact'])
    # land locations have EAI < 9e30
    allEAI[np.where(allEAI > 9e30)] = 0
    sumsq_eai_i, sum_eai_i = risk_sd_helper(allEAI)
    txx[:,:,i] = sumsq_eai_i
    tx[:,:,i] = sum_eai_i

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

# Location with max risk
# meanall[land_ind].max()
# np.argmax(meanall[land_ind]) # 266

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
# Bottom left figure

lon_land = data['lon']
lat_land = data['lat']

# Make 3 panel figure
fig, axs = plt.subplots(1, 3,
                        figsize=(11.12,7.34),
                        layout="constrained",
                        subplot_kw={'projection': ccrs.PlateCarree()})

# Plot of standard deviation risk
cp = axs[0].scatter(lon,lat,c=SDall,
                 norm=log_norm_sd,
                 s=11,
                 cmap='Purples')
# Create an inset axis for the colorbar
cax = inset_axes(axs[0], width="5%", height="40%", loc='upper right', borderpad=3.5)
cbar = plt.colorbar(cp, ax=axs[0], cax=cax, orientation='vertical')
cbar.set_label('Standard deviation of risk', labelpad=10)
cbar.ax.yaxis.set_label_position('left')
cax.text(3, 0.5, '(Annual person-days lost)', transform=cax.transAxes,
         ha='left', va='center', rotation=90, fontsize=9)
axs[0].set_title('Uncertainty of risk:\nvarying risk inputs', fontsize=14)
axs[0].coastlines()

# Plot number of decisions optimal in each cell
# Varying only risk
scatter = axs[1].scatter(lon_land,lat_land,
                           c=noptions_risk,s=12,cmap=dis_seq, norm=dis_seq_norm)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=legend_labs[i], 
           markersize=10, markerfacecolor=dis_seq_cols[i])
    for i in range(len(legend_labs))
]
axs[1].legend(handles=legend_elements)
axs[1].set_title('Uncertainty of decision:\nvarying risk inputs', fontsize=14)
axs[1].coastlines()
# Add a subplot for panel b
pie1 = inset_axes(axs[1], width="30%", height="30%", loc='upper right', borderpad = 1)
pie1.pie(counts_risk, colors = dis_seq_cols)

# Plot number of decisions optimal in each cell
# Varying risk and decision parameters
axs[2].set_xlabel('Longitude')
axs[2].set_ylabel('Latitude')
scatter = axs[2].scatter(lon_land,lat_land,c=noptions_riskdec,s=12,cmap=dis_seq, norm=dis_seq_norm)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=legend_labs[i], 
           markersize=10, markerfacecolor=dis_seq_cols[i])
    for i in range(len(legend_labs))
]
axs[2].legend(handles=legend_elements)
axs[2].set_title('Uncertainty of decision:\nvarying risk & decision inputs', fontsize=14)
axs[2].coastlines()

# Add a subplot for panel c
pie2 = inset_axes(axs[2], width="30%", height="30%", loc='upper right', borderpad = 1)
pie2.pie(counts_riskdec, colors = dis_seq_cols)

# Make sure plot (a) and (b) have the same limits as (c) and (d)
axs[0].set_xlim(axs[1].get_xlim())
axs[0].set_ylim(axs[1].get_ylim())

plt.savefig('../figures/poster-uncertainty2.png')
plt.show()

###################################################################
# Top right figure
# Map of decision sensitivity values for selected inputs

# Decision sensitivity data
max_dist_vals = np.load('../data/pawn_results/max_dist_vals_lhc200.npy')

# can be any data file
data = pd.read_csv("../data/decision_files_jit/OptimalDecision_ssp1_2deg_ChangeFactor_v1_53.78_v2_-3.804_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv")
lon = data['lon']
lat = data['lat']

# Map of decision sensitivity values for selected inputs
selected_indices = [0, 1, 5, 6]  # Selected i values
param_names = ['Choice of exposure model', 'Warming level', 'Cost per day of work', 'Cost of decision option 1']
fig, axes = plt.subplots(2, 2, figsize=(8, 6.95), subplot_kw={'projection': ccrs.PlateCarree()})

for ax, i in zip(axes.flat, selected_indices):
    cp = ax.scatter(lon, lat, c=max_dist_vals[i, :], vmin=0, vmax=0.5, s=5, cmap='Blues')
    ax.coastlines(linewidth=0.5)
    ax.set_title(param_names[selected_indices.index(i)], fontsize=14)
plt.tight_layout()

# Insert one large color bar
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=0.5))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.6)
cbar.set_label('Decision Sensitivity')

plt.savefig('../figures/poster-sensitivity2.png')
plt.show()