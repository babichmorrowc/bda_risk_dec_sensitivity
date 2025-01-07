
import numpy as np
import os
from matplotlib.colors import ListedColormap, BoundaryNorm

##########################################################################
# Import my functions
# Add the directory containing the functions to the Python path
# os.getcwd()
os.chdir("./code")
from bda_functions import *
from plotting_functions import *

##################################################################################################################
# Write a decision file for each of the 162 x n_samples total hypercube samples

# Set up range of risk options to vary
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = ["53.78", "54.5", "55.79"]
vuln2_opts = ["-4.597", "-4.1", "-3.804"]

# Read in Latin hypercube samples
n_samples = 200
X_n = np.loadtxt('../data/lat_hyp_samples_' + str(n_samples) + '.csv', delimiter=',')

# Loop over all combinations of risk x decision options
# Loop over risk parameters
for ssp in ssp_opts:
    for warm in warming_opts:
        # Exposure depends on SSP and SSP year (which comes from warming level)
        # Get SSP year to use based on warming level
        if warm == "2deg":
            ssp_year = 2041
        else:
            ssp_year = 2084
        # Get array of exposure in each cell
        Exp_array = get_Exp(input_data_path = '../data/',
                            ssp = ssp,
                            ssp_year = ssp_year)
        for cal in calibration_opts:
            for vuln1 in vuln1_opts:
                for vuln2 in vuln2_opts:
                    # String of risk inputs
                    risk_input_string = 'ssp'+ssp+'_'+warm+'_'+cal+'_v1_'+vuln1+'_v2_'+vuln2
                    print(risk_input_string)

                    # Get array of EAI
                    EAI_array = get_EAI(input_data_path = '../data/',
                                        data_source = cal,
                                        warming_level = warm,
                                        ssp = ssp,
                                        vp1 = vuln1,
                                        vp2 = vuln2)

                    ind, lat, lon = get_ind_lat_lon(Exp_array,
                                                    '../data/',
                                                    data_source = cal,
                                                    warming_level = warm,
                                                    ssp = ssp,
                                                    vp1 = vuln1,
                                                    vp2 = vuln2)
                    # Loop over decision parameters
                    for x in X_n:
                        # Extract decision parameters
                        # Columns of X are:
                        # cost_per_day, d2_1, d2_2, d3_1, d3_2, fin_weight
                        cost_per_day = x[0]
                        c_weight_cost = x[5]
                        cweights = np.array([float(c_weight_cost),1-float(c_weight_cost)])
                        dec_attributes = np.array([[0, 0, 5],
                                                   [x[1], x[2], 6],
                                                   [x[3], x[4], 4]])

                        # Write decision file
                        write_decision_file_jit(output_data_path = '../data/decision_files_' + str(n_samples) + '/',
                            overwrite = False,
                            ind = ind,
                            lat = lat,
                            lon = lon,
                            EAI = EAI_array,
                            Exp = Exp_array,
                            nd = 3,
                            decision_inputs = dec_attributes,
                            cost_per_day = cost_per_day,
                            cweights = cweights,
                            risk_input_string = risk_input_string)

###########################################################################
# Plot number of optimal decisions per cell

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = '../data/decision_files_' + str(n_samples) + '/'
file_list = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
n_files = len(file_list)

nloc_land = 1711
Y = np.empty(n_files*nloc_land).reshape(n_files,nloc_land)

# Loop over files in the folder
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    Y[i, :] = data['optimal_decision']


noptions = np.empty(nloc_land)
for i in range(1711):
    noptions[i] = len(np.unique(Y[:,i]))
np.unique(noptions, return_counts = True)

# List of all possible decisions
decision_options = np.unique(Y, return_counts = True)

# Plot number of decisions optimal in each cell
fig = plt.figure(figsize=(18,15))
lon_land = data['lon']
lat_land = data['lat']
ax1 = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cols = ListedColormap(['blue',
                    #    'orange',
                       'yellow'])
norm = BoundaryNorm([1,2,3], cols.N)
classes = ['1 optimal decision',
        #    '2 optimal decisions',
           '3 optimal decisions']
scatter = ax1.scatter(lon_land,lat_land,c=noptions,s=12,cmap=cols,norm=norm)
ax1.legend(handles=scatter.legend_elements()[0], labels=classes)
ax1.coastlines()
plt.show()

# Plot percentage of time that each decision was optimal in a cell

# Calculate frequencies of each decision per location
column_frequencies = {value: [] for value in decision_options[0]}
num_rows = Y.shape[0]

for col_index in range(Y.shape[1]):
    unique_values, value_counts = np.unique(Y[:, col_index], return_counts=True)
    # Create a dictionary to hold frequency counts for current column
    freq_dict = dict(zip(unique_values, value_counts))
    # Calculate frequencies for each decision option
    for value in decision_options[0]:
        if value in freq_dict:
            frequency = freq_dict[value] / num_rows
        else:
            frequency = 0.0  # Value not present, set frequency to 0
        
        column_frequencies[value].append(frequency)

frequencies_df = pd.DataFrame(column_frequencies)

fig = plt.figure(figsize=(18,9))
ax1 = plt.subplot(1,3,1,projection=ccrs.PlateCarree())
ax2 = plt.subplot(1,3,2,projection=ccrs.PlateCarree())
ax3 = plt.subplot(1,3,3,projection=ccrs.PlateCarree())
lon_land = data['lon']
lat_land = data['lat']

# Percent of time that decision 1 is optimal
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
scatter = ax1.scatter(lon_land,lat_land,c=frequencies_df[1],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax1,shrink=0.3)
cbar.set_label('% samples optimal')
ax1.coastlines()
ax1.title.set_text('Decision 1: % samples optimal')

# Percent of time that decision 2 is optimal
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
scatter = ax2.scatter(lon_land,lat_land,c=frequencies_df[2],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax2,shrink=0.3)
cbar.set_label('% samples optimal')
ax2.coastlines()
ax2.title.set_text('Decision 2: % samples optimal')

# Percent of time that decision 3 is optimal
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
scatter = ax3.scatter(lon_land,lat_land,c=frequencies_df[3],s=5,vmin=0,vmax=1,cmap='Greens')
cbar = plt.colorbar(scatter,ax=ax3,shrink=0.3)
cbar.set_label('% samples optimal')
ax3.coastlines()
ax3.title.set_text('Decision 3: % samples optimal')

plt.show()