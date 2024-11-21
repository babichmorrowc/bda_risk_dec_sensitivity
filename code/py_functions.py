# File of functions to call

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import cftime
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs

####################################################################################
# BDA FUNCTIONS
# Function to return the array of estimated annual impact (EAI)
def get_EAI(input_data_path,
            data_source,
            warming_level,
            ssp,
            vp1,
            vp2):
    # load in GAM samples of EAI
    gamsamples_file = input_data_path+data_source+'/GAMsamples_expected_annual_impact_data_'+data_source+'_WL'+warming_level+'_SSP'+ssp+'_vp1='+vp1+'_vp2='+vp2+'.nc'
    gamsamples = Dataset(gamsamples_file)
    EAI = np.array(gamsamples.variables['sim_annual_impact'])

    return EAI

# Function to return the array of number of people in each grid cell
def get_Exp(input_data_path,
            ssp,
            ssp_year):
    # need the number of ppl in each grid cell to calculate the total cost as input is 'cost per person'
    exposure_netcdf = Dataset(input_data_path+'UKSSPs/Employment_SSP'+ssp+'_12km_Physical.nc')
    units = getattr(exposure_netcdf['time'], 'units')
    calendar = getattr(exposure_netcdf['time'], 'calendar')
    dates = cftime.num2date(exposure_netcdf.variables['time'][:], units, calendar)
    year_to_index = {k.timetuple().tm_year: v for v, k in enumerate(dates)}
    # pick out the right year (SSP year)
    index = year_to_index[int(ssp_year)]
    Exp = np.array(exposure_netcdf.variables['employment'][index])

    return Exp

# Function to get indices of land locations and the corresponding arrays of latitude and longitude
# QUESTION: is this necessary to run every time/
def get_ind_lat_lon(Exp,
                    input_data_path,
                    data_source,
                    warming_level,
                    ssp,
                    vp1,
                    vp2):
    # load in GAM samples of EAI
    gamsamples_file = input_data_path+data_source+'/GAMsamples_expected_annual_impact_data_'+data_source+'_WL'+warming_level+'_SSP'+ssp+'_vp1='+vp1+'_vp2='+vp2+'.nc'
    gamsamples = Dataset(gamsamples_file)

    # find indices of land locations
    ind = np.where(Exp < 9e30)
    # only apply in land location
    lon = np.array(gamsamples.variables['exposure_longitude'])[ind[0],ind[1]]
    lat = np.array(gamsamples.variables['exposure_latitude'])[ind[0],ind[1]]

    return (ind, lat, lon)

# Function to find decision in a single cell
def decision_single_cell(ind,
                        index,
                        EAI,
                        Exp,
                        nd,
                        decision_inputs,
                        cost_per_day,
                        cweights):
    # state of nature (risk)
    xi = EAI[ind[0][index],ind[1][index],:]
     # If EAI in a region is < 0, we will set it to 0
    xi[np.where(xi < 0)] = 0
    # no. of people/jobs in each location
    ppl = Exp[ind[0][index],ind[1][index]]
    # If exposure in a region is 0, the decision should always be "do nothing"
    if ppl <= 0:
        opd = 1
        exp_util = np.full(nd, np.nan)
        util_scores = np.full([nd, 1000], np.nan)
        cost = np.full([nd, 1000], np.nan)
    # If there is exposure, find the Bayes optimal decision
    else:
        # calculate cost of each decision option for each of the 1000 GAM samples of risk
        cost = np.empty([nd,1000])
        for j in range(nd): # loop over number of decisions [do this in parallel?]
            for k in range(1000): # loop over GAM samples
                EAI_k = (10**xi[k] - 1)
                # cost outcome for decision j and sample k
                # QUESTION: Check that I'm converting from EAI to risk properly
                cost[j,k] = decision_inputs[j, 0]*ppl + cost_per_day*(1-decision_inputs[j, 1])*EAI_k
        
        # Meeting objectives scores
        meet_obs = decision_inputs[:,2]/10

        # Calculate the utility of each decision attribute (cost and meeting objectives), i.e. the value of different
        # values of each to the decision maker - here assuming a linear
        # utility but this could be elicited from the decision maker (i.e. how risk averse they are)
        util_cost = 1 + (-1 /cost.max()) * cost
        util_meet_obs = meet_obs

        # calculate the overall utility (value) of each decision option for each sample of risk
        util_scores = np.empty([nd, 1000])
        for j in range(nd):
            util_scores[j,:] = cweights[0] * util_cost[j,:] + cweights[1] * util_meet_obs[j]
        # find expected (mean) utility
        exp_util = np.empty(nd)
        for j in range(nd):
            exp_util[j] = np.mean(util_scores[j,:])
        #find which decision optimises the expected utility
        opd = np.where(exp_util == max(exp_util))[0] + 1 #(add one because python indexing starts at 0)

    # Return: optimal decision, expected utility, utility scores, cost
    return opd, exp_util, util_scores, cost

def write_decision_file(output_data_path,
                        overwrite,
                        ind,
                        lat,
                        lon,
                        EAI,
                        Exp,
                        nd,
                        decision_inputs,
                        cost_per_day,
                        cweights,
                        risk_input_string):
    # Name of output file
    output_file_string = '_d2_'+'_'.join(map(str, decision_inputs[1])) + '_d3_'+'_'.join(map(str, decision_inputs[2]))
    output_file_path_name = output_data_path+'OptimalDecision_'+risk_input_string+output_file_string+'.csv'
    
    # If overwrite = FALSE, check if a file of this name already exists
    if not overwrite:
        file_exists = os.path.isfile(output_file_path_name)
        if file_exists:
            print('File ' + output_file_path_name + ' already exists, skipping')
            return

    nloc = ind[0].shape[0]
    opd = np.empty(nloc)
    for i in range(nloc): # 1711 locations
        opd_i, exp_util_i, util_scores_i, cost_i = decision_single_cell(ind, i, EAI, Exp, nd, decision_inputs, cost_per_day, cweights)
        opd[i] = opd_i

    #save out optimal decision in each grid cell as a csv
    outputd = {'lon':lon,'lat':lat,'optimal_decision':opd}
    output=pd.DataFrame(outputd)

    output.to_csv(output_file_path_name)

    return(output)

####################################################################################
# PLOTTING FUNCTIONS

# Function to plot a location on the map given its index (for troubleshooting)
def plot_index(index):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    # Pick a random data file (all the lat / lons are the same)
    dat = pd.read_csv('../data/new_runs/vary_dec_attr/OptimalDecision_d2_80.0_20.0_50.0_7.0_d3_500.0_2.0_80.0_4.0.csv')
    # Plot all points in grey
    ax.scatter(dat['lon'], dat['lat'], c="grey", s=12)
    # Subset to desired index
    dat_ind = dat.loc[index]
    ax.scatter(dat_ind['lon'],dat_ind['lat'],c="red",s=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    return fig

# Function to plot decision outcomes on the map given the file name
def plot_decision_map(file_name):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    dat = pd.read_csv(file_name)
    cols = ListedColormap(['black','lawngreen','magenta'])
    classes = ['Do nothing','Modify working hours','Buy cooling equipment']
    scatter = ax.scatter(dat['lon'],dat['lat'],c=dat['optimal_decision'],cmap=cols,s=12)
    ax.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    return fig