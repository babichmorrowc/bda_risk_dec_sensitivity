# Code to test if the function modifications are working

import os
import time

#####################################################################################################################
# Import functions defined in py_functions.py
# Add the directory containing py_functions.py to the Python path
# os.getcwd()
os.chdir("./code")
from py_functions import *

#####################################################################################################################
# Look at risk data
# EAI
# Risk parameters used
data_source = "UKCP_BC"
warming_level = "2deg"
ssp = "2"
vp1 = "54.5"
vp2 = "-4.1"

# Get EAI
EAI_array = get_EAI('../data/', data_source, warming_level, ssp, vp1, vp2)
EAI_array.shape # (110, 83, 1000)
# What exactly am I looking at here?

# Look at exposure in those cells
Exp_array = get_Exp('../data/', ssp, 2041)
Exp_array.shape # (110, 83)
Exp_array.min()

ind, lat, lon = get_ind_lat_lon(Exp_array,
                       '../data/', data_source, warming_level, ssp, vp1, vp2)
# ind consists of two arrays, each of length 1711
# lat and lon are both length 1711


# # Figure out how to filter 0 EAI
# EAI_array_1 = EAI_array[ind[0][910],ind[1][910],:]
# EAI_array_1[np.where(EAI_array_1 < 0)] = 0


# Set decision-related parameters to be held constant
# Cost per day and cost vs. meeting objectives weight
cost_per_day = 200
c_weight_cost = 1 # Going all financial cost
# c_weight_cost = 0.8 # default value
#criterion weights
cweights = [float(c_weight_cost),1-float(c_weight_cost)]

dec_attributes = np.array([[0, 0, 5],
                           [300, 0.5, 7],
                           [600, 0.8, 4]])

# Check on decision in a single cell
london_ind = 241

start = time.time()
lon_opd, lon_exp_util, lon_util_scores, lon_cost = decision_single_cell(ind,
                        index = london_ind,
                        EAI = EAI_array,
                        Exp = Exp_array,
                        nd = 3,
                        decision_inputs = dec_attributes,
                        cost_per_day = cost_per_day,
                        cweights = cweights)
end = time.time()
print(end - start)

lon_opd 
lon_exp_util


# Check to see if I'm fixing the issues in my problem points
# Time the function
start = time.time()
test_decision = write_decision_file(output_data_path = "../data/test/",
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
                    risk_input_string = 'test')
end = time.time()
print(end - start)

# Check london decision
test_decision.iloc[[241]]

# Check unique decisions
np.unique(test_decision['optimal_decision'], return_counts=True)

test_plot = plot_decision_map("../data/test/OptimalDecision_test_d2_300.0_0.5_7.0_d3_600.0_0.8_4.0.csv")
plt.show()




