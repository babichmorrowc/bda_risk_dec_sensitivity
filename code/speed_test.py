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
end = time.time()
print("Time taken for writing all decision files with jit:", end - start)