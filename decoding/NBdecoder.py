import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split


class NBdecoder2D:
    def __init__(self, n_cells, n_bins_x, n_bins_y, box_range=None):
        self.n_cells = n_cells
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.ratemaps = np.zeros((n_cells, n_bins_x, n_bins_y))
        self.occupancy = np.zeros((n_bins_x, n_bins_y))

        self.box_range = None  # should have shape (2,2)

    def fit_ratemaps(self, X, Y, sigma=0):
        ''' args: X (n_times x n_cells), spikecounts, Y (n_times x 2) trajectory '''

        if self.box_range is None:
            box_range = [[np.nanmin(Y[:, 0]), np.nanmax(Y[:, 0])],
                         [np.nanmin(Y[:, 1]), np.nanmax(Y[:, 1])]]

        else:
            box_range = self.box_range

        occupancy = binned_statistic_2d(Y[:, 0], Y[:, 1], np.ones(len(Y)),
                                        statistic='count',
                                        bins=[self.n_bins_x, self.n_bins_y],
                                        range=box_range)[0]

        for i, cell_X in enumerate(X.T):
            spikemap = binned_statistic_2d(Y[:, 0], Y[:, 1], cell_X,
                                           statistic='sum',
                                           bins=[self.n_bins_x, self.n_bins_y],
                                           range=box_range)[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratemap = spikemap/occupancy
            ratemap[np.isnan(ratemap)] = 0

            if sigma > 0:
                ratemap = gaussian_filter(ratemap, sigma)
            self.ratemaps[i, :, :] = ratemap

        return

    def fit_occupancy(self, Y, sigma=0):
        ''' args: Y (n_times x 2) trajectory '''

        if self.box_range is None:
            box_range = np.asarray([[np.nanmin(Y[:, 0]), np.nanmax(Y[:, 0])],
                                    [np.nanmin(Y[:, 1]), np.nanmax(Y[:, 1])]])

        else:
            box_range = self.box_range

        self.occupancy = binned_statistic_2d(Y[:, 0], Y[:, 1], np.ones(len(Y)),
                                             statistic='count',
                                             bins=[self.n_bins_x,
                                                   self.n_bins_y],
                                             range=box_range)[0]
        if sigma > 0:
            self.occupancy = gaussian_filter(self.occupancy, sigma)

        return

    def fit(self, X, Y, sigma=0):
        ''' args: X (n_times x n_cells), spikecounts, Y (n_times x 2) trajectory '''

        if self.box_range is None:
            self.box_range = [[np.nanmin(Y[:, 0]), np.nanmax(Y[:, 0])],
                              [np.nanmin(Y[:, 1]), np.nanmax(Y[:, 1])]]

        self.occupancy = binned_statistic_2d(Y[:, 0], Y[:, 1], np.ones(len(Y)),
                                             statistic='count',
                                             bins=[self.n_bins_x,
                                                   self.n_bins_y],
                                             range=self.box_range)[0]

        for i, cell_X in enumerate(X.T):
            spikemap = binned_statistic_2d(Y[:, 0], Y[:, 1], cell_X,
                                           statistic='sum',
                                           bins=[self.n_bins_x, self.n_bins_y],
                                           range=self.box_range)[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratemap = spikemap/self.occupancy
            ratemap[np.isnan(ratemap)] = 0

            if sigma > 0:
                ratemap = gaussian_filter(ratemap, sigma)
            self.ratemaps[i, :, :] = ratemap

        if sigma > 0:
            self.occupancy = gaussian_filter(self.occupancy, sigma)

        return

    def predict_posterior(self, X, epsilon=pow(10, -15), use_prior=True):
        ratemaps = self.ratemaps.reshape(
            self.n_cells, self.n_bins_x*self.n_bins_y)
        if use_prior:
            prior = self.occupancy.flatten() / sum(self.occupancy.flatten())
            log_posteriors = X @ np.log(ratemaps+epsilon) - \
                np.sum(ratemaps, axis=0) + np.log(prior + epsilon)
        else:
            log_posteriors = X @ np.log(ratemaps+epsilon) - \
                np.sum(ratemaps, axis=0)

        posteriors = np.exp(log_posteriors)
        posteriors = posteriors / sum(posteriors.flatten())
        return posteriors.reshape(posteriors.shape[0], self.n_bins_x, self.n_bins_y)

    def predict_position(self, X, use_prior=True):

        if self.box_range is None:
            print('ERROR: unknown box range. The model has not been fitted')
        else:
            bin_x = np.linspace(
                self.box_range[0][0], self.box_range[0][1], self.n_bins_x)
            bin_y = np.linspace(
                self.box_range[1][0], self.box_range[1][1], self.n_bins_y)

            posteriors = self.predict_posterior(X, use_prior=use_prior)
            predicted_position_bin = np.asarray([np.unravel_index(
                np.argmax(posterior), posterior.shape) for posterior in posteriors])
            predicted_position = [[bin_x[i[0]], bin_y[i[1]]]
                                  for i in predicted_position_bin]

        return np.asarray(predicted_position)

    def control_mse(self, Y_test):
        bin_x = np.linspace(
            self.box_range[0][0], self.box_range[0][1], self.n_bins_x)
        bin_y = np.linspace(
            self.box_range[1][0], self.box_range[1][1], self.n_bins_y)
        prior_position_bin = np.unravel_index(
            np.argmax(self.occupancy), self.occupancy.shape)
        prior_y = [bin_x[prior_position_bin[0]], bin_y[prior_position_bin[1]]]
        control_mse = (np.mean([distance(true_y, prior_y)
                       for true_y in Y_test]))
        return control_mse


# UTILITY FUNCTIONS

def shuffled_control_mse(X_train, Y_train, X_test, Y_test,
                         n_bins_x, n_bins_y, box_range,
                         sigma_ratemaps, use_prior=True, 
                         n_shuffles=10,min_shift=20):

    mse = []
    for _ in range(n_shuffles):
        # shuffle xtrain
        X_train_shuff = shuffle_spikes(X_train,min_shift=min_shift)
        shuff_decoder = NBdecoder2D(n_cells=X_train_shuff.shape[-1],
                                    n_bins_x=n_bins_x,
                                    n_bins_y=n_bins_y,
                                    box_range=box_range)
        shuff_decoder.fit(X_train_shuff, Y_train, sigma=sigma_ratemaps)
        Y_pred = shuff_decoder.predict_position(X_test, use_prior=use_prior)
        mse.append(mean_square_error(Y_pred, Y_test))

    avg_mse = np.nanmean(mse)
    std_mse = np.nanstd(mse)

    return avg_mse, std_mse


def bin_spikes(spikes, times):
    # builds spike count matrix
    start_time = times[0]
    end_time = times[-1]
    n_bins = len(times)
    X = np.empty((n_bins, len(spikes)))
    time_bins = np.linspace(start_time, end_time, n_bins+1)
    for i, cell_spikes in enumerate(spikes):
        X[:, i] = np.histogram(cell_spikes, bins=time_bins)[0]

    return X


def shuffle_spikes(X, min_shift=20):
    X_shuff = np.zeros(X.shape)
    for i in range(X.shape[-1]):
        X_shuff[:,i] = np.roll(X[:,i], np.random.choice(
            np.arange(min_shift, X.shape[-1])))
    return X_shuff


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
    x, y = position[0, :], position[1, :]
    # inteprolate x
    nans, t = find_nans(x)
    if sum(nans) > 0:
        x[nans] = np.interp(t(nans), t(~nans), x[~nans])

    # inteprolate y
    nans, t = find_nans(y)
    if sum(nans) > 0:
        y[nans] = np.interp(t(nans), t(~nans), y[~nans])

    return np.asarray([x, y])


def distance(pos1, pos2):
    return np.sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)


def mean_square_error(Y_pred, Y_true):
    mse = np.mean([distance(y_pred, y_true)
                  for y_pred, y_true in zip(Y_pred, Y_true)])
    return mse


def compute_occupancy(Y, n_bins, box_range=None, sigma=0):
    '''Arguments:
        Y : 2 x n_samples trajectory array
        n_bins : [n_bins_x,n_bins_y] bin in x and y
        box_range : 2 x 2 matrix with box range
        sigma : smoothing gaussian stdev

       Returns:
        occupancy array
    '''
    if box_range is None:
        box_range = np.asarray([[np.nanmin(Y[:, 0]), np.nanmax(Y[:, 0])],
                                [np.nanmin(Y[:, 1]), np.nanmax(Y[:, 1])]])

    else:
        box_range = box_range

    occupancy = binned_statistic_2d(Y[:, 0], Y[:, 1], np.ones(len(Y)),
                                    statistic='count',
                                    bins=n_bins,
                                    range=box_range)[0]
    if sigma > 0:
        occupancy = gaussian_filter(occupancy, sigma)

    return occupancy


def compute_occupancy_similarity(Y1, Y2, n_bins, box_range=None, sigma=0):
    occupancy1 = compute_occupancy(
        Y1, n_bins, box_range=box_range, sigma=sigma)
    occupancy2 = compute_occupancy(
        Y2, n_bins, box_range=box_range, sigma=sigma)
    similarity = np.corrcoef(occupancy1, occupancy2)[0, 1]
    return similarity
