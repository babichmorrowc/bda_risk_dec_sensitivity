
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

