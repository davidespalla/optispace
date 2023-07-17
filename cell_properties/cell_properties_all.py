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
animals = [2, 3, 4, 5]

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

session_dict = {2: ['1Rectangle', '2Circle'],
                3: ['1Square', '2Circle'],
                4: ['1Circle', '2Square', '3Circle'],
                5: ['1Circle', '2Square']}


metadata = pd.read_csv('../../SleepData_Matteo/Open_field/metadata.csv')
pixel_to_cm = metadata.pixel_to_cm.values[0]

Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(ratemaps_folder).mkdir(parents=True, exist_ok=True)

for animal in animals:
    animal_metadata = metadata[metadata.animal == animal]
    cell_properties = pd.DataFrame()
    sessions = session_dict[animal]
    for session in sessions:
        print(f'Computing animal {animal}, session {session}')
        # import data

        path = f'../../SleepData_Matteo/Open_field/AH{animal}/{session}'
        mat = io.loadmat(path+'/spike_times.mat')
        spike_times = mat['spike_times'][0]
        spike_times = [i.flatten() for i in spike_times]

        mat = io.loadmat(path+'/trajectorypos.mat')
        position = mat['trajectorypos']

        mat = io.loadmat(path+'/pos_times.mat')
        pos_times = mat['pos_times'][0]

        # %% interpolate and smooth
        position = fs.interpolate_nans(position)

        if smooth_position:
            position[:, 0] = ndimage.gaussian_filter1d(
                position[:, 0], sigma_pos)
            position[:, 1] = ndimage.gaussian_filter1d(
                position[:, 1], sigma_pos)

        # %%AVERAGE RATE AND INCLUSION FLAG
        print('Calculating average rates ...')
        avg_rates = np.asarray([len(i) for i in spike_times])/pos_times[-1]
        cell_properties[f'avg_rate_{session}'] = avg_rates
        rate_mask = [1 if (x < firingh_th_high and x > firing_th_low)
                     else 0 for x in avg_rates]
        cell_properties[f'rate_include_{session}'] = rate_mask

        # %% CALCULATE AND SAVE RATEMAPS
        print('Calculating and saving ratemaps ... ')

        x_min = animal_metadata[animal_metadata.session == int(
            session[0])].x_min.values[0]
        x_max = animal_metadata[animal_metadata.session == int(
            session[0])].x_max.values[0]
        y_min = animal_metadata[animal_metadata.session == int(
            session[0])].y_min.values[0]
        y_max = animal_metadata[animal_metadata.session == int(
            session[0])].y_max.values[0]

        box_range = [[x_min, x_max], [y_min, y_max]]
        xbins = int((x_max - x_min)*pixel_to_cm/ratemap_bin_size)
        ybins = int((y_max - y_min)*pixel_to_cm/ratemap_bin_size)
        ratemap_bins = [xbins, ybins]

        ratemaps = []
        for cell in range(len(spike_times)):
            rm = fs.compute_ratemap(
                spike_times[cell], position, pos_times, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)
            ratemaps.append(rm)
        ratemaps = np.asarray(ratemaps)
        np.save(ratemaps_folder +
                f'/ratemaps_animal{animal}_{session}.npy', ratemaps)

        # %% CALCULATE SPATIAL INFO AND PLACE CELL FLAG
        print('Calculating spatial info and place cell flags ...')

        len_session = pos_times[-1]  # length of session in seconds

        occupancy = fs.compute_occupancy(
            position, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)
        occupancy_prob = occupancy/sum(occupancy.flatten())
        np.save(ratemaps_folder +
                f'/occupancy_animal{animal}_{session}.npy', occupancy)

        spatial_info = []
        place_cell_flag = []
        for cell in range(len(ratemaps)):

            s_info = fs.skaggs_info_perspike(
                ratemaps[cell], occupancy_prob)
            spatial_info.append(s_info)

            shuff_spatial_info = []
            for shuf in range(n_shuffles):

                random_shift = random.uniform(min_shift, len_session)

                shifted_spikes = [(s+random_shift) %
                                  len_session for s in spike_times[cell]]

                shuff_ratemap = fs.compute_ratemap(
                    shifted_spikes, position, pos_times, nbins=ratemap_bins, range=box_range, sigma=sigma_ratemaps)

                shuff_spatial_info.append(fs.skaggs_info_perspike(
                    shuff_ratemap, occupancy_prob))

            place_cell_treshold = np.percentile(
                shuff_spatial_info, percentile_th)
            if s_info > place_cell_treshold:
                place_cell_flag.append(True)
            else:
                place_cell_flag.append(False)

            print(f'Processed cell {cell+1}/{len(ratemaps)}')

        print(
            f'Done, found {sum(place_cell_flag)}/{len(place_cell_flag)} place cells')

        cell_properties[f'spatial_info_{session}'] = spatial_info
        cell_properties[f'place_cell_flag_{session}'] = place_cell_flag

    #  SAVE CELL PROPERTIES
    cell_properties.to_csv(
        output_folder + f'/cell_properties_animal{animal}.csv')
    print('place cells properties saved')
