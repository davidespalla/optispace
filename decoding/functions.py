import numpy as np
from scipy.ndimage import gaussian_filter


def spike_positions_2d(spike_times, position, pos_times):

    spike_x = np.interp(spike_times, pos_times, position[:, 0])
    spike_y = np.interp(spike_times, pos_times, position[:, 1])
    return spike_x, spike_y


def bin_spikes(spikes, times, bin_width):
    ''' 
    spikes - list of lists of spike timestamps
    times - times of the position
    bin_width -  bin width in same unit of times
    '''

    start_time = times[0]
    end_time = times[-1]
    n_bins = int(np.ceil((end_time-start_time)/bin_width))
    time_bins = np.linspace(start_time, end_time, n_bins)
    spikecount_array = np.empty((len(spikes), n_bins))
    for cell in range(len(spikes)):
        spikecount_array[cell, :] = np.histogram(
            spikes[cell], bins=time_bins)[0]
    return spikecount_array


def compute_ratemap(spike_times, positions, pos_times, nbins=20, spatial_range=None, sigma=None):
    spike_positions = spike_positions_2d(spike_times, positions, pos_times)
    if not spatial_range:
        spatial_range = [[min(positions[:, 0]), max(positions[:, 0])], [
            min(positions[:, 1]), max(positions[:, 1])]]
    occupancy = np.histogram2d(positions[:, 0], positions[:, 1], bins=[
                               nbins, nbins], range=spatial_range)[0]
    spikemap = np.histogram2d(spike_positions[0], spike_positions[1], bins=[
                              nbins, nbins], range=spatial_range)[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = spikemap / occupancy

    rate_map[np.logical_or(np.isnan(rate_map), np.isinf(rate_map))] = 0

    if sigma:
        occupancy = gaussian_filter(occupancy, sigma)
        rate_map = gaussian_filter(rate_map, sigma)

    #rate_map[np.where(occupancy==0)] = np.nan
    return rate_map


def compute_occupancy(positions, nbins=20, spatial_range=None, sigma=None):
    if not spatial_range:
        spatial_range = [[min(positions[:, 0]), max(positions[:, 0])], [
            min(positions[:, 1]), max(positions[:, 1])]]

    occupancy = np.histogram2d(positions[:, 0], positions[:, 1], bins=[
                               nbins, nbins], range=spatial_range)[0]

    if sigma:
        occupancy = gaussian_filter(occupancy, sigma)

    return occupancy


def compute_spikemap(spike_times, positions, pos_times, nbins=[20, 20], spatial_range=None, sigma=None):
    spike_positions = spike_positions_2d(spike_times, positions, pos_times)
    if not spatial_range:
        spatial_range = [[min(positions[:, 0]), max(positions[:, 0])], [
            min(positions[:, 1]), max(positions[:, 1])]]
    spikemap = np.histogram2d(
        spike_positions[0], spike_positions[1], bins=nbins, range=spatial_range)[0]
    if sigma:
        spikemap = gaussian_filter(spikemap, sigma)

    return spikemap


def build_cost_matrix(hist_shape, bin_width=1):
    '''
    takes shape of histogram in bins and linear bin width
    returns cost matrix M, with ditances in the same units as bin width
    '''
    nbx = hist_shape[0]
    nby = hist_shape[1]
    [bin_x, bin_y] = np.meshgrid(
        np.arange(nbx)*bin_width, np.arange(nby)*bin_width)
    bin_x = bin_x.flatten()
    bin_y = bin_y.flatten()
    N = nbx*nby
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            x1, y1 = bin_x[i], bin_y[i]
            x2, y2 = bin_x[j], bin_y[j]
            M[i, j] = np.sqrt((x2-x1)**2+(y2-y1)**2)
            M[j, i] = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return M


def build_cost_matrix_square(hist_shape):
    # OBS: only works for sauqre histograms
    nbx = hist_shape[0]
    nby = hist_shape[1]
    [bin_x, bin_y] = np.meshgrid(
        np.linspace(0, 1, nbx), np.linspace(0, 1, nby))
    bin_x = bin_x.flatten()
    bin_y = bin_y.flatten()
    N = nbx*nby
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            x1, y1 = bin_x[i], bin_y[i]
            x2, y2 = bin_x[j], bin_y[j]
            M[i, j] = np.sqrt((x2-x1)**2+(y2-y1)**2)
            M[j, i] = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return M/max(M.flatten())


def compute_peak_distance(ratemap1, ratemap2):
    '''
    takes two ratemaps and returns distance between peaks, in normalized bin difference
    '''
    x1, y1 = np.where(ratemap1 == ratemap1.max())[
        0][0], np.where(ratemap1 == ratemap1.max())[1][0]
    x2, y2 = np.where(ratemap2 == ratemap2.max())[
        0][0], np.where(ratemap2 == ratemap2.max())[1][0]

    dist = np.sqrt((x2/ratemap2.shape[0] - x1/ratemap1.shape[0])
                   ** 2 + (y2/ratemap2.shape[0] - y1/ratemap1.shape[0])**2)
    return dist


def skaggs_info_perspike(rate_map, occupancy_prob, epsilon=pow(10, -15)):
    rate_map = rate_map.flatten()
    occupancy_prob = occupancy_prob.flatten()
    avg_rate = np.mean(rate_map)
    if avg_rate > 0:
        return sum(rate_map*np.log2((rate_map+epsilon)/avg_rate)*occupancy_prob)/avg_rate
    else:
        return np.nan


def find_nans(y):
    """
    returns nan mask and nan indices in array
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(position):
    '''
    Takes position array with shape (len_session,2)
    Returns position array with nan values linearly interpolated
    '''
    x, y = position[:, 0], position[:, 1]
    # inteprolate x
    nans, t = find_nans(x)
    if sum(nans) > 0:
        x[nans] = np.interp(t(nans), t(~nans), x[~nans])

    # inteprolate y
    nans, t = find_nans(y)
    if sum(nans) > 0:
        y[nans] = np.interp(t(nans), t(~nans), y[~nans])

    return np.asarray([x, y]).T
