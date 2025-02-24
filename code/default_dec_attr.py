# Code to find a default set of decision parameters
# Which will produce some cells with only one optimal decision
# When varying the risk-related parameters

import os
import time

##########################################################################
# Import my functions
# Add the directory containing the functions to the Python path
# os.getcwd()
os.chdir("./code")
from bda_functions import *
from plotting_functions import *

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
cweights = [c_weight_cost,1-c_weight_cost]

###########################################################################
# Plot loss functions

# Define a sequence of numbers for x axis
x = np.linspace(0,6,200)

# Define inputs
nd = 3

# Set up a 3 column figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

p = 100 # 100 jobs in location
cost_100 = np.empty([nd,len(x)])
for j in range(nd):
    for k in range(len(x)):
        cost_100[j, k] = dec_attributes[j, 0] * p + cost_per_day * (1 - dec_attributes[j, 1]) * (10 ** x[k])
log_cost_100 = np.log10(cost_100)
ax1.plot(x,log_cost_100[0,:],label='Do nothing',color='black')
ax1.plot(x,log_cost_100[1,:],label='Modify working hours',color='lawngreen')
ax1.plot(x,log_cost_100[2,:],label='Buy cooling equipment',color='magenta')
ax1.set_xlabel('EAI in grid cell (log base 10)')
ax1.set_ylabel('Cost, £ (log base 10)')
ax1.legend()
ax1.title.set_text('100 jobs')

p = 1000 # 1000 jobs in location
cost_1000 = np.empty([nd,len(x)])
for j in range(nd):
    for k in range(len(x)):
        cost_1000[j, k] = dec_attributes[j, 0] * p + cost_per_day * (1 - dec_attributes[j, 1]) * (10 ** x[k])
log_cost_1000 = np.log10(cost_1000)
ax2.plot(x,log_cost_1000[0,:],label='Do nothing',color='black')
ax2.plot(x,log_cost_1000[1,:],label='Modify working hours',color='lawngreen')
ax2.plot(x,log_cost_1000[2,:],label='Buy cooling equipment',color='magenta')
ax2.set_xlabel('EAI in grid cell (log base 10)')
ax2.set_ylabel('Cost, £ (log base 10)')
ax2.legend()
ax2.title.set_text('1000 jobs')

p = 10000 # 10000 jobs in location
cost_10000 = np.empty([nd,len(x)])
for j in range(nd):
    for k in range(len(x)):
        cost_10000[j, k] = dec_attributes[j, 0] * p + cost_per_day * (1 - dec_attributes[j, 1]) * (10 ** x[k])
log_cost_10000 = np.log10(cost_10000)
ax3.plot(x,log_cost_10000[0,:],label='Do nothing',color='black')
ax3.plot(x,log_cost_10000[1,:],label='Modify working hours',color='lawngreen')
ax3.plot(x,log_cost_10000[2,:],label='Buy cooling equipment',color='magenta')
ax3.set_xlabel('EAI in grid cell (log base 10)')
ax3.set_ylabel('Cost, £ (log base 10)')
ax3.legend()
ax3.title.set_text('10000 jobs')

plt.show()


###########################################################################
# # Test in London to ensure there is only one optimal decision
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

# # How many decisions are optimal?
# print("In London, there are", len(np.unique(lon_results_df[['opt_dec']])), "optimal decision(s)")

# # See counts of types of decisions
# np.unique(lon_results_df[['opt_dec']], return_counts = True)

# # See rows where the decision was 2 for troubleshooting
# # lon_results_df[lon_results_df['opt_dec'] == 2]

###########################################################################
# Plot the optimal decision in each cell using all the default values

# Get EAI
def_EAI_array = get_EAI('../data/', def_calibration, def_warming, def_ssp, def_vuln1, def_vuln2)

# Get exposure
def_Exp_array = get_Exp('../data/', def_ssp, 2041)

def_ind, def_lat, def_lon = get_ind_lat_lon(def_Exp_array,
                       '../data/', def_calibration, def_warming, def_ssp, def_vuln1, def_vuln2)

# Time the function
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
print(end - start)

# Check number of locations in which each decision is optimal
np.unique(test_decision['optimal_decision'], return_counts=True)

# Plot the optimal decision in each location using all the defaults
test_plot = plot_decision_map("../data/test/OptimalDecision_defaults_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv")
plt.show()


###########################################################################
# Start writing decision files for all combinations of risk parameters (at default decision parameters)
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
                    write_decision_file_jit(output_data_path = '../data/decision_files/',
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


