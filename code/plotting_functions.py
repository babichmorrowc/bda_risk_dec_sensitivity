import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import pandas as pd

####################################################################################
# PLOTTING FUNCTIONS

# Function to plot a location on the map given its index (for troubleshooting)
def plot_index(index):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    # Pick a random data file (all the lat / lons are the same)
    dat = pd.read_csv('../data/decision_files_jit/OptimalDecision_ssp1_2deg_ChangeFactor_v1_53.78_v2_-3.804_d2_250.0_0.4_6.0_d3_600.0_0.8_4.0.csv')
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
def plot_decision_map(file_name, ax=None):
    dat = pd.read_csv(file_name)
    cols = ListedColormap(['gray','lawngreen','magenta'])
    classes = ['Do nothing','Modify working hours','Buy cooling equipment']
    # Create figure and ax only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig = ax.figure  # grab the figure from the provided axes
    scatter = ax.scatter(dat['lon'],dat['lat'],c=dat['optimal_decision'],cmap=cols,s=12)
    ax.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    return ax