import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs

###########################################################################
# Varying risk inputs ONLY
# Plot number of optimal decisions per cell

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = './data/decision_files_jit/'
file_list = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
n_files = len(file_list)

nloc_land = 1711
Y_risk = np.empty(n_files*nloc_land).reshape(n_files,nloc_land)

# Loop over files in the folder
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    Y_risk[i, :] = data['optimal_decision']

noptions_risk = np.empty(nloc_land)
for i in range(1711):
    noptions_risk[i] = len(np.unique(Y_risk[:,i]))
np.unique(noptions_risk, return_counts = True)

# List of all possible decisions
decision_options = np.unique(Y_risk, return_counts = True)

# Plot number of decisions optimal in each cell
fig = plt.figure(figsize=(18,15))
lon_land = data['lon']
lat_land = data['lat']
ax1 = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cols = ListedColormap(['blue','orange','yellow'])
classes = ['1 optimal decision','2 optimal decisions','3 optimal decisions']
scatter = ax1.scatter(lon_land,lat_land,c=noptions_risk,s=12,cmap=cols)
ax1.legend(handles=scatter.legend_elements()[0], labels=classes)
ax1.coastlines()
plt.show()

# # Plot percentage of time that each decision was optimal in a cell

# # Calculate frequencies of each decision per location
# column_frequencies = {value: [] for value in decision_options[0]}
# num_rows = Y.shape[0]

# for col_index in range(Y.shape[1]):
#     unique_values, value_counts = np.unique(Y[:, col_index], return_counts=True)
#     # Create a dictionary to hold frequency counts for current column
#     freq_dict = dict(zip(unique_values, value_counts))
#     # Calculate frequencies for each decision option
#     for value in decision_options[0]:
#         if value in freq_dict:
#             frequency = freq_dict[value] / num_rows
#         else:
#             frequency = 0.0  # Value not present, set frequency to 0
        
#         column_frequencies[value].append(frequency)

# frequencies_df = pd.DataFrame(column_frequencies)

# fig = plt.figure(figsize=(18,9))
# ax1 = plt.subplot(1,3,1,projection=ccrs.PlateCarree())
# ax2 = plt.subplot(1,3,2,projection=ccrs.PlateCarree())
# ax3 = plt.subplot(1,3,3,projection=ccrs.PlateCarree())
# lon_land = data['lon']
# lat_land = data['lat']

# # Percent of time that decision 1 is optimal
# ax1.set_xlabel('Longitude')
# ax1.set_ylabel('Latitude')
# scatter = ax1.scatter(lon_land,lat_land,c=frequencies_df[1],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax1,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax1.coastlines()
# ax1.title.set_text('Decision 1: % samples optimal')

# # Percent of time that decision 2 is optimal
# ax2.set_xlabel('Longitude')
# ax2.set_ylabel('Latitude')
# scatter = ax2.scatter(lon_land,lat_land,c=frequencies_df[2],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax2,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax2.coastlines()
# ax2.title.set_text('Decision 2: % samples optimal')

# # Percent of time that decision 3 is optimal
# ax3.set_xlabel('Longitude')
# ax3.set_ylabel('Latitude')
# scatter = ax3.scatter(lon_land,lat_land,c=frequencies_df[3],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax3,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax3.coastlines()
# ax3.title.set_text('Decision 3: % samples optimal')

# plt.show()

###########################################################################
# Varying risk AND decision inputs
# Plot number of optimal decisions per cell

n_samples = 200

# For all runs in the decision_files folder
# Get the list of all files in the folder
folder_path = './data/decision_files_' + str(n_samples) + '/'
file_list = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
n_files = len(file_list)

nloc_land = 1711
Y_riskdec = np.empty(n_files*nloc_land).reshape(n_files,nloc_land)

# Loop over files in the folder
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    Y_riskdec[i, :] = data['optimal_decision']


noptions_riskdec = np.empty(nloc_land)
for i in range(1711):
    noptions_riskdec[i] = len(np.unique(Y_riskdec[:,i]))
np.unique(noptions_riskdec, return_counts = True)

# List of all possible decisions
decision_options = np.unique(Y_riskdec, return_counts = True)

# Plot number of decisions optimal in each cell
fig = plt.figure(figsize=(18,15))
lon_land = data['lon']
lat_land = data['lat']
ax1 = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cols = ListedColormap(['blue',
                    #    'orange',
                       'yellow'])
# norm = BoundaryNorm([1,2,3], cols.N)
classes = ['1 optimal decision',
        #    '2 optimal decisions',
           '3 optimal decisions']
scatter = ax1.scatter(lon_land,lat_land,c=noptions_riskdec,s=12,cmap=cols)
ax1.legend(handles=scatter.legend_elements()[0], labels=classes)
ax1.coastlines()
plt.show()

# # Plot percentage of time that each decision was optimal in a cell

# # Calculate frequencies of each decision per location
# column_frequencies = {value: [] for value in decision_options[0]}
# num_rows = Y.shape[0]

# for col_index in range(Y.shape[1]):
#     unique_values, value_counts = np.unique(Y[:, col_index], return_counts=True)
#     # Create a dictionary to hold frequency counts for current column
#     freq_dict = dict(zip(unique_values, value_counts))
#     # Calculate frequencies for each decision option
#     for value in decision_options[0]:
#         if value in freq_dict:
#             frequency = freq_dict[value] / num_rows
#         else:
#             frequency = 0.0  # Value not present, set frequency to 0
        
#         column_frequencies[value].append(frequency)

# frequencies_df = pd.DataFrame(column_frequencies)

# fig = plt.figure(figsize=(18,9))
# ax1 = plt.subplot(1,3,1,projection=ccrs.PlateCarree())
# ax2 = plt.subplot(1,3,2,projection=ccrs.PlateCarree())
# ax3 = plt.subplot(1,3,3,projection=ccrs.PlateCarree())
# lon_land = data['lon']
# lat_land = data['lat']

# # Percent of time that decision 1 is optimal
# ax1.set_xlabel('Longitude')
# ax1.set_ylabel('Latitude')
# scatter = ax1.scatter(lon_land,lat_land,c=frequencies_df[1],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax1,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax1.coastlines()
# ax1.title.set_text('Decision 1: % samples optimal')

# # Percent of time that decision 2 is optimal
# ax2.set_xlabel('Longitude')
# ax2.set_ylabel('Latitude')
# scatter = ax2.scatter(lon_land,lat_land,c=frequencies_df[2],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax2,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax2.coastlines()
# ax2.title.set_text('Decision 2: % samples optimal')

# # Percent of time that decision 3 is optimal
# ax3.set_xlabel('Longitude')
# ax3.set_ylabel('Latitude')
# scatter = ax3.scatter(lon_land,lat_land,c=frequencies_df[3],s=5,vmin=0,vmax=1,cmap='Greens')
# cbar = plt.colorbar(scatter,ax=ax3,shrink=0.3)
# cbar.set_label('% samples optimal')
# ax3.coastlines()
# ax3.title.set_text('Decision 3: % samples optimal')

# plt.show()

###########################################################################
# Bar graph comparing number of optimal decisions
# Varying only risk- vs. varying risk- and decision-related attributes

counts_risk = np.array([np.sum(noptions_risk == num) for num in decision_options[0]])
counts_riskdec = np.array([np.sum(noptions_riskdec == num) for num in decision_options[0]])

bar_width = 0.4

fig, ax = plt.subplots()
ax.bar(decision_options[0]-bar_width/2, counts_risk, width=bar_width, label='Varying risk only')
ax.bar(decision_options[0]+bar_width/2, counts_riskdec, width=bar_width, label='Varying risk and decision inputs')
ax.set_xlabel('Number of optimal decisions')
ax.set_ylabel('Number of cells')
ax.set_xticks(decision_options[0])
ax.legend()
plt.show()