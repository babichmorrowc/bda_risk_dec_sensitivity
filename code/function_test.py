# Code to test if the function modifications are working

import sys
import os

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
cost_per_day = 100
c_weight_cost = 1 # Going all financial cost
#criterion weights
cweights = [float(c_weight_cost),1-float(c_weight_cost)]

dec_attributes = np.array([[0, 0, 0, 5],
                           [100, 20, 50, 7],
                           [500, 2, 80, 4]])

# Check to see if I'm fixing the issues in my problem points
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
                    cweights = cweights)

test_plot = plot_decision_map("../data/test/OptimalDecision_d2_100_20_50_7_d3_500_2_80_4.csv")
plt.show()

# What's going on in the points where decisions still look odd?
test_index_scotland = plot_index(1322)
plt.show()
test_index_ni = plot_index(1042)
plt.show()

# Get EAI just in the land locations
EAI_masked = EAI_array[ind[0], ind[1], :]
EAI_masked.shape # (1711, 1000)

# Get exposure just in the land locations
Exp_masked = Exp_array[ind[0], ind[1]]
Exp_masked.shape # (1711,)

test_decision.iloc[[1322]]
Exp_masked[1322] # 0.070594636545941
EAI_masked[1322]

Exp_masked[1042] # 0.2091511276629873
EAI_masked[1042]

plt.hist(cost[0,:], bins=50, alpha=0.5, label='Do nothing')
plt.hist(cost[1,:], bins=50, alpha=0.5, label='Modify working hours')
plt.hist(cost[2,:], bins=50, alpha=0.5, label='Buy cooling equipment')
plt.legend()
plt.show()



