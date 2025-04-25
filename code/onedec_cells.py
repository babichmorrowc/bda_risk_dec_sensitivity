import os
import numpy as np
import pandas as pd

os.chdir("./code")
from bda_functions import *

# Get number of optimal decisions in each cell
# When varying both risk and decision parameters
n_samples = 200
nloc_land = 1711

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

# Look into locations where only one decision is optimal
onedec_locs = [i for i in range(len(noptions_riskdec)) if noptions_riskdec[i] == 1]

# Get exposure values for these cells
onedec_exp = []
ssp_opts = ["1", "2", "5"]
for ssp in ssp_opts:
    for ssp_year in [2041, 2084]: # for 2 and 4 degrees, respectively
        exposure_data = get_Exp('../data/', ssp = ssp, ssp_year = ssp_year)
        exposure_data = exposure_data[np.where(exposure_data < 9e30)] # filter to land
        for loc in onedec_locs:
            exp_loc = exposure_data[loc]
            onedec_exp.append([loc, ssp, ssp_year, exp_loc])
onedec_exp = pd.DataFrame(onedec_exp)
onedec_exp.columns = ['loc', 'ssp', 'year', 'exposure']
onedec_exp