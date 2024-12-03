# Speed test
# File to check on the speed of the functions

import os
import time

##########################################################################
# Import functions defined in py_functions.py
# Add the directory containing py_functions.py to the Python path
# os.getcwd()
os.chdir("./code")
from py_functions import *

##########################################################################
# Set up range of risk options to vary
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = ["53.78", "54.5", "55.79"]
vuln2_opts = ["-4.597", "-4.1", "-3.804"]

# Set up default risk options
def_calibration = "UKCP_BC"
def_warming = "2deg"
def_ssp = "2"
def_vuln1 = "54.5"
def_vuln2 = "-4.1"

##########################################################################
# Set up potential decision attributes
# Cost per day of work lost:
cost_per_day = 200
# Cost attributes of each of the 3 decisions + objective scores
dec_attributes = np.array([[0, 0, 5],
                           [250, 0.4, 6],
                           [600, 0.8, 4]])
# Relative weighting of priorities
c_weight_cost = 0.8 
cweights = np.array([float(c_weight_cost),1-float(c_weight_cost)])

##########################################################################
# # Check the speed of decision_single_cell WITHOUT jit
# # Test in London 
lon_ind = 241

# # Create empty list to store optimal decisions in each location
# lon_results = []

# # Loop over all combinations of risk parameters
# start = time.time()
# for ssp in ssp_opts:
#     for warm in warming_opts:
#         # Exposure depends on SSP and SSP year (which comes from warming level)
#         # Get SSP year to use based on warming level
#         if warm == "2deg":
#             ssp_year = 2041
#         else:
#             ssp_year = 2084
#         # Get array of exposure in each cell
#         Exp_array = get_Exp(input_data_path = '../data/',
#                             ssp = ssp,
#                             ssp_year = ssp_year)
#         for cal in calibration_opts:
#             for vuln1 in vuln1_opts:
#                 for vuln2 in vuln2_opts:
#                     # Get array of EAI
#                     EAI_array = get_EAI(input_data_path = '../data/',
#                                         data_source = cal,
#                                         warming_level = warm,
#                                         ssp = ssp,
#                                         vp1 = vuln1,
#                                         vp2 = vuln2)

#                     ind, lat, lon = get_ind_lat_lon(Exp_array,
#                                                     '../data/',
#                                                     data_source = cal,
#                                                     warming_level = warm,
#                                                     ssp = ssp,
#                                                     vp1 = vuln1,
#                                                     vp2 = vuln2)

#                     # Get decision in the cell
#                     lon_opd, lon_exp_util, lon_util_scores, lon_cost = decision_single_cell(
#                         ind = ind,
#                         index = lon_ind,
#                         EAI = EAI_array,
#                         Exp = Exp_array,
#                         nd = 3,
#                         decision_inputs = dec_attributes,
#                         cost_per_day = cost_per_day,
#                         cweights = cweights
#                     )
#                     lon_results.append({'ssp': ssp,
#                                         'warm': warm,
#                                         'cal': cal,
#                                         'vuln1': vuln1,
#                                         'vuln2': vuln2,
#                                         'opt_dec': lon_opd[0]})
# end = time.time()
# print("Time taken for decision_single_cell without jit:", end - start)
# # 9.39619517326355 seconds
# # For 162 iterations

##########################################################################
def_Exp_array = get_Exp('../data/', def_ssp, 2041)
def_EAI_array = get_EAI('../data/', def_calibration, def_warming, def_ssp, def_vuln1, def_vuln2)
def_ind, def_lat, def_lon = get_ind_lat_lon(def_Exp_array,
                       '../data/', def_calibration, def_warming, def_ssp, def_vuln1, def_vuln2)

# Time the first run of decision_single_cell WITH jit
# This includes compilation time
start = time.time()
lon_opd, lon_exp_util, lon_util_scores, lon_cost = decision_single_cell_jit(
                        ind = def_ind,
                        index = lon_ind,
                        EAI = def_EAI_array,
                        Exp = def_Exp_array,
                        nd = 3,
                        decision_inputs = dec_attributes,
                        cost_per_day = cost_per_day,
                        cweights = cweights
                    )
end = time.time()
print("Time taken for decision_single_cell with jit INCLUDING compilation:", end - start)
# 0.021577119827270508

# Time all runs of decision_single_cell WITH jit
# Create empty list to store optimal decisions in each location
lon_results = []
# Loop over all combinations of risk parameters
start = time.time()
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

                    # Get decision in the cell
                    lon_opd, lon_exp_util, lon_util_scores, lon_cost = decision_single_cell_jit(
                        ind = ind,
                        index = lon_ind,
                        EAI = EAI_array,
                        Exp = Exp_array,
                        nd = 3,
                        decision_inputs = dec_attributes,
                        cost_per_day = cost_per_day,
                        cweights = cweights
                    )
                    lon_results.append({'ssp': ssp,
                                        'warm': warm,
                                        'cal': cal,
                                        'vuln1': vuln1,
                                        'vuln2': vuln2,
                                        'opt_dec': lon_opd[0]})
end = time.time()
print("Time taken for decision_single_cell with jit:", end - start)
# 3.6322 seconds
# For 162 iterations

##########################################################################
# Time the first run of decision_single_cell WITH jit AND parallel
# This includes compilation time
start = time.time()
lon_opd, lon_exp_util, lon_util_scores, lon_cost = decision_single_cell_jit_parallel(
                        ind = def_ind,
                        index = lon_ind,
                        EAI = def_EAI_array,
                        Exp = def_Exp_array,
                        nd = 3,
                        decision_inputs = dec_attributes,
                        cost_per_day = cost_per_day,
                        cweights = cweights
                    )
end = time.time()
print("Time taken for decision_single_cell with jit / parallel INCLUDING compilation:", end - start)
# 2.1169559955596924

# Parallel diagnostics
decision_single_cell_jit_parallel.parallel_diagnostics(level=4)

# Time all runs of decision_single_cell WITH jit
# Create empty list to store optimal decisions in each location
par_results = []
# Loop over all combinations of risk parameters
start = time.time()
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

                    # Get decision in the cell
                    par_opd, par_exp_util, par_util_scores, par_cost = decision_single_cell_jit_parallel(
                        ind = ind,
                        index = lon_ind,
                        EAI = EAI_array,
                        Exp = Exp_array,
                        nd = 3,
                        decision_inputs = dec_attributes,
                        cost_per_day = cost_per_day,
                        cweights = cweights
                    )
                    par_results.append({'ssp': ssp,
                                        'warm': warm,
                                        'cal': cal,
                                        'vuln1': vuln1,
                                        'vuln2': vuln2,
                                        'opt_dec': lon_opd[0]})
end = time.time()
print("Time taken for decision_single_cell with jit and parallel:", end - start)
# 4.7327046394348145 seconds
# For 162 iterations

# Test that the results are the same
lon_results == par_results

##########################################################################
# Time writing an entire file without jit

start = time.time()
test_decision = write_decision_file(output_data_path = "../data/test/",
                    overwrite = True,
                    ind = def_ind,
                    lat = def_lat,
                    lon = def_lon,
                    EAI = def_EAI_array,
                    Exp = def_Exp_array,
                    nd = 3,
                    decision_inputs = dec_attributes,
                    cost_per_day = cost_per_day,
                    cweights = cweights,
                    risk_input_string = 'defaults')
end = time.time()
print("Time taken for write_decision_file without jit:", end - start)
# 8.554831266403198 seconds

##########################################################################
# Time writing an entire file with jit
start = time.time()
test_decision = write_decision_file_jit(output_data_path = "../data/test/",
                    overwrite = True,
                    ind = def_ind,
                    lat = def_lat,
                    lon = def_lon,
                    EAI = def_EAI_array,
                    Exp = def_Exp_array,
                    nd = 3,
                    decision_inputs = dec_attributes,
                    cost_per_day = cost_per_day,
                    cweights = cweights,
                    risk_input_string = 'defaults')
end = time.time()
print("Time taken for write_decision_file_jit using jit:", end - start)
# 0.21112728118896484 WOOOOOOHOOOOOOOO

# Plot the optimal decision in each location using all the defaults
test_plot = plot_decision_map("../data/test/OptimalDecision_defaults_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv")
plt.show()

##########################################################################
# Time writing an entire file with jit and parallel
start = time.time()
test_decision = write_decision_file_jit_parallel(output_data_path = "../data/test/",
                    overwrite = True,
                    ind = def_ind,
                    lat = def_lat,
                    lon = def_lon,
                    EAI = def_EAI_array,
                    Exp = def_Exp_array,
                    nd = 3,
                    decision_inputs = dec_attributes,
                    cost_per_day = cost_per_day,
                    cweights = cweights,
                    risk_input_string = 'defaults')
end = time.time()
print("Time taken for write_decision_file_jit using jit and parallel:", end - start)
# 0.516772985458374

# Parallel diagnostics
write_decision_file_jit_parallel.parallel_diagnostics(level=4)

# Plot the optimal decision in each location using all the defaults
test_plot = plot_decision_map("../data/test/OptimalDecision_defaults_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv")
plt.show()

##########################################################################
# Time writing a file for all combinations of risk parameters
# Start writing decision files for all combinations of risk parameters (at default decision parameters)
# Loop over all combinations of risk parameters
start = time.time()
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

                    # Write decision file
                    write_decision_file_jit(output_data_path = '../data/decision_files_jit/',
                        overwrite = True,
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
end = time.time()
print("Time taken for writing all decision files with jit:", end - start)
# 15.856331825256348
# For 162 files

##########################################################################
# Time writing a file for all combinations of risk parameters
# Start writing decision files for all combinations of risk parameters (at default decision parameters)
# Loop over all combinations of risk parameters
start = time.time()
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

                    # Write decision file
                    write_decision_file_jit_parallel(output_data_path = '../data/decision_files_jit/',
                        overwrite = True,
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
end = time.time()
print("Time taken for writing all decision files with jit and parallel:", end - start)
# 14.94858694076538
# For 162 files

###########################################################################
# Plot number of optimal decisions per cell

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = '../data/decision_files_jit/'
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
# cols = ListedColormap(['green','orange','red'])
cols = ListedColormap(['blue','orange','yellow'])
classes = ['1 optimal decision','2 optimal decisions','3 optimal decisions']
scatter = ax1.scatter(lon_land,lat_land,c=noptions,s=12,cmap=cols)
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