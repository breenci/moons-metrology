"""
Module for aligning point clouds using the Kabsch-Umeyama algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
from src.transform.spatial import matrix_transform

def kabsch_umeyama(A, B, scale=True):
    """Apply the Kabsch Uneyama algorithm to align two point clouds

    :param A: reference point cloud
    :type A: numpy.ndarray
    :param B: point cloud to be aligned
    :type B: numpy.ndarray
    :param scale: allow scaling in tranformation, defaults to True
    :type scale: bool, optional
    :return: transformation matrix for alignment of B to A
    :rtype: numpy.ndarray
    """

    # see https://web.stanford.edu/class/cs273/refs/umeyama.pdf
    n, m = A.shape

    # get mean vector of A and B
    A_cntr = np.mean(A, axis=0)
    B_cntr = np.mean(B, axis=0)

    # calculate the variance of A
    A_var = np.mean(np.linalg.norm(A - A_cntr, axis=1) ** 2)

    # get the covariance matrix of A and B and do SVD 
    covar_mat = np.matmul((A - A_cntr).T, (B - B_cntr)) / n
    U, D, VT = np.linalg.svd(covar_mat)

    # S = identity matrix if Det(U)*Det(V) = 1
    # S = Diag(1,...,1,-1) if Det(U)*Det(V) = -1
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    # define rotation matrix (R), scaling factor (c) and translation (t) 
    R = np.matmul(np.matmul(U, S), VT)
    c = A_var / np.trace(np.matmul(np.diag(D), S))
    
    # bring all together to create augmented transformation matrix
    if scale is True:
        t = A_cntr - np.matmul(c * R, B_cntr)
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = c * R
        trans_mat[:3, 3] = t

    if scale is False:
        t = A_cntr - np.matmul(R, B_cntr)
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = R
        trans_mat[:3, 3] = t

    return trans_mat


def align_measurements(ref, unal, targ_ids, scale=False):
    """Aligns reference and unaligned datasets using Kabsch-Umeyama algorithm

    :param ref: reference dataset
    :type ref: numpy.ndarray
    :param unal: unaligned dataset
    :type unal: numpy.ndarray
    :param targ_ids: ids of points to be used in alignment
    :type targ_ids: numpy.ndarray
    :param scale: allow scling in transformation, defaults to False
    :type scale: bool, optional
    :return: aligned coordinates
    :rtype: numpy.ndarray
    """
    # load ref id and xyz
    # get targets from input ids
    ref_ids = ref[:, 0]

    # load each unaligned data set and align to ref using KU
    unal_ids = unal[:, 0]

    # Only use reference targets that are detected in both datasets
    # Use ids to find these targets
    common_ids, ref_idxs, unal_idxs = np.intersect1d(ref_ids, unal_ids, 
                                                     return_indices=True)

    # Mask to select targets
    common_targs_mask = np.isin(common_ids, targ_ids)

    # get position of targets in reference and unaligned datasets
    ref_targs = ref[ref_idxs[common_targs_mask]]
    unal_targs = unal[unal_idxs[common_targs_mask]]

    # get transformation matrix for the alignment using KU
    al_mat = kabsch_umeyama(ref_targs[:, 1:4], unal_targs[:, 1:4], scale=scale)

    # Do alignment and return aligned coords
    al_coords = matrix_transform(unal[:,1:4], al_mat)

    return al_coords


def align_from_file(ref_fn, unal_fn, targ_ids):

    ref_arr = np.loadtxt(ref_fn)
    unal_arr = np.loadtxt(unal_fn)

    al_coords = align_measurements(ref_arr, unal_arr, targ_ids)

    return al_coords