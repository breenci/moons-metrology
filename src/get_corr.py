"""
Module for finding corresponding points between two datasets
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.transform.spatial import do_transform


def rescale(mask1, mask2, ref1, ref2):
    """Rescale mask2 so that the distance between two reference points is
    the same in mask1 and mask2.
    """

    # calculate ref distance in both coordinate systems
    mask1_dist = dist_calc(mask1[ref1[0], :], mask1[ref1[1], :])
    mask2_dist = dist_calc(mask2[ref2[0], :], mask2[ref2[1], :])

    # Apply scaling factor and return rescaled mask2 data
    scale = mask1_dist/mask2_dist
    rescaled_data = scale*mask2

    return rescaled_data


def dist_calc(pnt1, pnt2):
    """Calculate the distance between 2 points"""

    # convert points to numpy arrays
    pnt1_arr = np.array(pnt1)
    pnt2_arr = np.array(pnt2)

    dist = np.sqrt(np.sum((pnt1_arr - pnt2_arr)**2))

    return dist


def match_near_nbrs(data1, data2, tol):
    """Use nearest neighbours analysis to find the indices of corresponding
    points between the two datasets.
    """

    # Stack datasets for nearest neighbours analysis
    full_data = np.vstack((data1, data2))

    # Nearest neighbours analysis
    nbrs = NearestNeighbors(n_neighbors=1,
                            algorithm='ball_tree').fit(full_data)
    distances, indices = nbrs.kneighbors()

    # get indices data2
    data1_idx = np.arange(0, len(data1))
    data2_idx = indices[data1_idx] - len(data1)

    # reject points where NN is too far away. Indicates a missing point.
    distances1 = distances[data1_idx]
    idx1_out = data1_idx[distances1[:, 0] < tol]
    idx2_out = data2_idx[distances1 < tol]

    return idx1_out, idx2_out


def get_corr(mask1, mask2, ref1, ref2):
    '''Finds corresponding points between the two mask datasets. Rough aligns
    mask2 to mask1 using common reference points and then uses nearest
    neighbours analysis to find indices of corresponding points'''

    # Transform mask data into coordinate system defined by common reference
    # points to rough align
    mask1_idxs = np.arange(0, len(mask1))
    RA_mask1 = do_transform(mask1, mask1_idxs, ref1[0], ref1[1])

    mask2_idxs = np.arange(0, len(mask2))
    trans_mask2 = do_transform(mask2, mask2_idxs, ref2[0], ref2[1])

    # rescale mask2 to complete rough alignment
    RA_mask2 = rescale(RA_mask1, trans_mask2, ref1, ref2)

    # NN analysis
    ind1, ind2 = match_near_nbrs(RA_mask1, RA_mask2, .3)

    return ind1, ind2


if __name__ == '__main__':
    # load datasets
    cam_mask = np.loadtxt('mask_test/mask_AC_01.txt')
    met_mask = np.loadtxt('mask_test/transformed_coords_mask1_15092021.txt')

    # pad and reflect AC camera image in y axis
    cmask_pad = np.hstack((cam_mask, np.zeros((len(cam_mask[:, 1]), 1))))
    cmask_pad[:, 1] = -1 * cmask_pad[:, 1]

    a, b = get_corr(met_mask, cmask_pad, (25, 11), (13, 0))