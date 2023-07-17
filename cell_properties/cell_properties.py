'''
calculate and save the followig cell properies:
- spatial info
- place cells flag
'''
# %%
from random import random
import functions as fs
import numpy as np
from scipy import ndimage
from scipy import io
from scipy import interpolate
import pandas as pd
import random
from pathlib import Path
# %% PARAMETERS
animal = 2

session_a = '1Rectangle'
session_b = '2Circle'

metadata = pd.read_csv('../../SleepData_Matteo/Open_field/metadata.csv')
metadata = metadata[metadata.animal == animal]
pixel_to_cm = metadata.pixel_to_cm.values[0]


smooth_position = True
sigma_pos = 2

ratemap_bin_size = 1  # in cm
sigma_ratemaps = 1

firingh_th_high = 10  # in Hz
firing_th_low = 0.1

n_shuffles = 100
min_shift = 5  # in seconds
percentile_th = 99  # percentile threshold for place cell flagging

output_folder = '../processed_data/cell_properties'
ratemaps_folder = '../processed_data/ratemaps'

# %% Globals
cell_properties = pd.DataFrame()
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(ratemaps_folder).mkdir(parents=True, exist_ok=True)

# %% SESSION A
print(f'Analysis of session A: {session_a}')
# import data

path = f'../../SleepData_Matteo/Open_field/AH{animal}/{session_a}'
mat = io.loadmat(path+'/spike_times.mat')
spike_times_a = mat['spike_times'][0]
spike_times_a = [i.flatten() for i in spike_times_a]

mat = io.loadmat(path+'/trajectorypos.mat')
position_a = mat['trajectorypos']

mat = io.loadmat(path+'/pos_times.mat')
pos_times_a = mat['pos_times'][0]

# %% interpolate and smooth
position_a = fs.interpolate_nans(position_a)

if smooth_position:
    position_a[:, 0] = ndimage.gaussian_filter1d(position_a.T[0], sigma_pos)
    position_a[:, 1] = ndimage.gaussian_filter1d(position_a[:, 1], sigma_pos)

# %%AVERAGE RATE AND INCLUSION FLAG
print('Calculating average rates ...')
avg_rates_a = np.asarray([len(i) for i in spike_times_a])/pos_times_a[-1]
cell_properties['avg_rate_A'] = avg_rates_a
rate_mask = [1 if (x < firingh_th_high and x > firing_th_low)
             else 0 for x in avg_rates_a]
cell_properties['rate_include_A'] = rate_mask


# %% CALCULATE AND SAVE RATEMAPS
print('Calculating and saving ratemaps ... ')

x_min = metadata[metadata.session == int(session_a[0])].x_min.values[0]
x_max = metadata[metadata.session == int(session_a[0])].x_max.values[0]
y_min = metadata[metadata.session == int(session_a[0])].y_min.values[0]
y_max = metadata[metadata.session == int(session_a[0])].y_max.values[0]

box_range = [[x_min, x_max], [y_min, y_max]]
xbins = int((x_max - x_min)*pixel_to_cm/ratemap_bin_size)
ybins = int((y_max - y_min)*pixel_to_cm/ratemap_bin_size)
ratemap_bins = [xbins, ybins]

ratemaps_a = []
for cell in range(len(spike_times_a)):
    rm = fs.compute_ratemap(
        spike_times_a[cell], position_a, pos_times_a, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)
    ratemaps_a.append(rm)
ratemaps_a = np.asarray(ratemaps_a)
np.save(ratemaps_folder +
        f'/ratemaps_animal{animal}_session_{session_a}.npy', ratemaps_a)

# %% CALCULATE SPATIAL INFO AND PLACE CELL FLAG
print('Calculating spatial info and place cell flags ...')

len_session = pos_times_a[-1]  # length of session in seconds

occupancy_a = fs.compute_occupancy(
    position_a, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)
occupancy_prob_a = occupancy_a/sum(occupancy_a.flatten())

spatial_info_a = []
place_cell_flag_a = []
for cell in range(len(ratemaps_a)):

    spatial_info = fs.skaggs_info_perspike(
        ratemaps_a[cell], occupancy_prob_a)
    spatial_info_a.append(spatial_info)

    shuff_spatial_info = []
    for shuf in range(n_shuffles):

        random_shift = random.uniform(min_shift, len_session)

        shifted_spikes = [(s+random_shift) %
                          len_session for s in spike_times_a[cell]]

        shuff_ratemap = fs.compute_ratemap(
            shifted_spikes, position_a, pos_times_a, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)

        shuff_spatial_info.append(fs.skaggs_info_perspike(
            shuff_ratemap, occupancy_prob_a))

    place_cell_treshold = np.percentile(shuff_spatial_info, percentile_th)
    if spatial_info > place_cell_treshold:
        place_cell_flag_a.append(True)
    else:
        place_cell_flag_a.append(False)

    print(f'Processed cell {cell+1}/{len(ratemaps_a)}')

print(
    f'Done, found {sum(place_cell_flag_a)}/{len(place_cell_flag_a)} place cells')


cell_properties['spatial_info_A'] = spatial_info_a
cell_properties['place_cell_flag_A'] = place_cell_flag_a

#############################################################

# %% SESSION B
print(f'Analysis of session B: {session_b}')
# import data
path = f'../../SleepData_Matteo/Open_field/AH{animal}/{session_b}'
mat = io.loadmat(path+'/spike_times.mat')
spike_times_b = mat['spike_times'][0]
spike_times_b = [i.flatten() for i in spike_times_b]

mat = io.loadmat(path+'/trajectorypos.mat')
position_b = mat['trajectorypos']

mat = io.loadmat(path+'/pos_times.mat')
pos_times_b = mat['pos_times'][0]

# %% interpolate and smooth
position_b = fs.interpolate_nans(position_b)

if smooth_position:
    position_b[:, 0] = ndimage.gaussian_filter1d(position_b.T[0], sigma_pos)
    position_b[:, 1] = ndimage.gaussian_filter1d(position_b[:, 1], sigma_pos)

# %%AVERAGE RATE AND INCLUSION FLAG
print('Calculating average rates ...')
avg_rates_b = np.asarray([len(i) for i in spike_times_b])/pos_times_b[-1]
cell_properties['avg_rate_B'] = avg_rates_b
rate_mask = [1 if (x < firingh_th_high and x > firing_th_low)
             else 0 for x in avg_rates_b]
cell_properties['rate_include_B'] = rate_mask

# %% CALCULATE AND SAVE RATEMAPS
print('Calculating and saving ratemaps ... ')

x_min = metadata[metadata.session == int(session_b[0])].x_min.values[0]
x_max = metadata[metadata.session == int(session_b[0])].x_max.values[0]
y_min = metadata[metadata.session == int(session_b[0])].y_min.values[0]
y_max = metadata[metadata.session == int(session_b[0])].y_max.values[0]

box_range = [[x_min, x_max], [y_min, y_max]]
xbins = int((x_max - x_min)*pixel_to_cm/ratemap_bin_size)
ybins = int((x_max - x_min)*pixel_to_cm/ratemap_bin_size)
ratemap_bins = [xbins, ybins]

ratemaps_b = []
for cell in range(len(spike_times_b)):
    rm = fs.compute_ratemap(
        spike_times_b[cell], position_b, pos_times_b, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)
    ratemaps_b.append(rm)
ratemaps_b = np.asarray(ratemaps_b)
np.save(ratemaps_folder +
        f'/ratemaps_animal{animal}_session_{session_b}.npy', ratemaps_b)

# %% CALCULATE SPATIAL INFO AND PLACE CELL FLAG
print('Calculating spatial info and place cell flags ...')

len_session = pos_times_b[-1]  # length of session in seconds

occupancy_b = fs.compute_occupancy(
    position_b, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)
occupancy_prob_b = occupancy_b/sum(occupancy_b.flatten())

spatial_info_b = []
place_cell_flag_b = []
for cell in range(len(ratemaps_b)):

    spatial_info = fs.skaggs_info_perspike(
        ratemaps_b[cell], occupancy_prob_b)
    spatial_info_b.append(spatial_info)

    shuff_spatial_info = []
    for shuf in range(n_shuffles):

        random_shift = random.uniform(min_shift, len_session)

        shifted_spikes = [(s+random_shift) %
                          len_session for s in spike_times_b[cell]]

        shuff_ratemap = fs.compute_ratemap(
            shifted_spikes, position_b, pos_times_b, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)

        shuff_spatial_info.append(fs.skaggs_info_perspike(
            shuff_ratemap, occupancy_prob_b))

    place_cell_treshold = np.percentile(shuff_spatial_info, percentile_th)
    if spatial_info > place_cell_treshold:
        place_cell_flag_b.append(True)
    else:
        place_cell_flag_b.append(False)

    print(f'Processed cell {cell+1}/{len(ratemaps_b)}')

print(
    f'Done, found {sum(place_cell_flag_b)}/{len(place_cell_flag_b)} place cells')


cell_properties['spatial_info_B'] = spatial_info_b
cell_properties['place_cell_flag_B'] = place_cell_flag_b

# %% SAVE CELL PROPERTIES
cell_properties.to_csv(output_folder + f'/cell_properties_animal{animal}.csv')
print('place cells properties saved')


# %%
