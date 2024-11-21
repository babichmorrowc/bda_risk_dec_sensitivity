# Code to find a default set of decision parameters
# Which will produce some cells with only one optimal decision
# When varying the risk-related parameters

import os

#####################################################################################################################
# Import functions defined in py_functions.py
# Add the directory containing py_functions.py to the Python path
# os.getcwd()
os.chdir("./code")
from py_functions import *

##########################################################################
# Set up risk options to vary
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = ["53.78", "54.5", "55.79"]
vuln2_opts = ["-4.597", "-4.1", "-3.804"]

##########################################################################
# Set up potential decision attributes
# Cost per day of work lost:
cost_per_day = 200
# Cost attributes of each of the 3 decisions + objective scores
dec_attributes = np.array([[0, 0, 5],
                           [300, 0.4, 7],
                           [600, 0.8, 4]])
# Relative weighting of priorities
c_weight_cost = 0.8 
cweights = [float(c_weight_cost),1-float(c_weight_cost)]

###########################################################################
# # Test in London
# # lon = -0.0920587, lat = 51.471443
# lon_ind = 241

# # Create empty list to store optimal decisions in each location
# lon_results = []

# # Loop over all combinations of risk parameters
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

# # Convert to dataframe
# lon_results_df = pd.DataFrame(lon_results)

# # See counts of types of decisions
# np.unique(lon_results_df[['opt_dec']], return_counts = True)

# # See rows where the decision was 2
# lon_results_df[lon_results_df['opt_dec'] == 2]

###########################################################################
# Start writing decision files
# Loop over all combinations of risk parameters
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
                    write_decision_file(output_data_path = '../data/decision_files/',
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
folder_path = '../data/decision_files/'
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
# Only 11 cells with 3 optimal decisions

# List of all possible decisions
decision_options = np.unique(Y, return_counts = True)
# Only 47 2's

fig = plt.figure(figsize=(18,15))
lon_land = data['lon']
lat_land = data['lat']
ax1 = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cols = ListedColormap(['green','orange','red'])
classes = ['1 optimal decision','2 optimal decisions','3 optimal decisions']
scatter = ax1.scatter(lon_land,lat_land,c=noptions,s=12,cmap=cols)
ax1.legend(handles=scatter.legend_elements()[0], labels=classes)
ax1.coastlines()
plt.show()
