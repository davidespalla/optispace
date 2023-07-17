import numpy as np
import ot


class DistanceCalculator():
    def __init__(self, ratemaps):
        self.n_cells = len(ratemaps)
        self.ratemaps_shape = ratemaps[0].shape
        self.ratemaps = ratemaps
        self.cost_matrix = build_cost_matrix(self.ratemaps_shape)
        # builds normalized and flattened ratemaps
        norm_ratemaps = []
        for i in ratemaps:
            norm_ratemaps.append(i.ravel()/sum(i.ravel()))
        self.norm_ratemaps = norm_ratemaps

        self.ot_dists = np.zeros((self.n_cells, self.n_cells))
        self.peak_dists = np.zeros((self.n_cells, self.n_cells))
        self.corr_dists = np.zeros((self.n_cells, self.n_cells))

    def compute_ot_dists(self, verbose=True):
        for i in range(self.n_cells-1):
            if verbose:
                print(f"computing row {i+1}/{self.n_cells}")
            for j in range(i+1, self.n_cells):
                _, log = ot.emd(self.norm_ratemaps[i], self.norm_ratemaps[j],
                                self.cost_matrix, log=True)
                ot_cost = log['cost']
                self.ot_dists[i, j] = self.ot_dists[j, i] = ot_cost
        return

    def compute_corr_dists(self):
        for i in range(self.n_cells):
            for j in range(i, self.n_cells):
                corr_dist = 1 - np.corrcoef(
                    self.norm_ratemaps[i], self.norm_ratemaps[j])[0, 1]
                self.corr_dists[i, j] = self.corr_dists[j, i] = corr_dist
        return

    def compute_peak_dists(self):
        for i in range(self.n_cells-1):
            for j in range(i+1, self.n_cells):
                peak_dist = compute_peak_distance(
                    self.ratemaps[i], self.ratemaps[j])
                self.peak_dists[i, j] = self.peak_dists[j, i] = peak_dist
        return

    def compute_all_dists():
        return


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
