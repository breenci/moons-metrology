import numpy as np
import matplotlib.pyplot as plt
from src.get_corr import get_corr
from src.coord_transform2 import matrix_transform


def kabsch_umeyama(A, B, scale=True):
    """Apply the Kabsch Uneyama algorithm"""

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
    """Aligns point clouds with reference"""
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


if __name__ == '__main__':
    # # load datasets
    # cam_mask = np.loadtxt('data/mask_test/mask_AC_01.txt')
    # met_mask = np.loadtxt('data/mask_test/transformed_coords_mask1_15092021.txt')
    # #met_mask[:,0] = 0

    # cmask_pad = np.hstack((cam_mask, np.zeros((len(cam_mask[:,1]), 1))))

    # a,b = get_corr(met_mask, cmask_pad, (25,11), (13,24))

    # srtd_cmask = cmask_pad[b]
    # srtd_mmask = met_mask[a]

    # t_mat = kabsch_umeyama(srtd_mmask, srtd_cmask)
    # print(t_mat)
    # trans_cmask = matrix_transform(srtd_cmask, t_mat)


    # diff = np.abs(srtd_mmask - trans_cmask)
    # rms = np.sqrt(np.mean(diff**2, axis=0))
    # rms_pnts = np.sqrt(np.mean(diff**2, axis=1))
    
    # fig1, ax1 = plt.subplots()
    # ax1.scatter(np.arange(len(diff[:,1])), diff[:,1], c='r', marker='+', label='Y')
    # ax1.scatter(np.arange(len(diff[:,1])), diff[:,2], c='b', s=10, label='Z')
    # ax1.scatter(np.arange(len(diff[:,1])), diff[:,0], c='g', s=10, label='X')
    # ax1.legend()
    # ax1.axhline(rms[2], c='b')
    # ax1.axhline(rms[1], c='r')
    # ax1.axhline(rms[0], c='g')
    # ax1.set_ylabel('RMS Error (mm)')
    # ax1.set_xlabel('Dot No.')
    # plt.show()

    # fig2, ax = plt.subplots()
    # scat = ax.scatter(srtd_mmask[:,1], srtd_mmask[:,2], c=rms_pnts)
    # cb = plt.colorbar(scat)

    n=10000
    
    al_100 = align_from_file('data/temp_tests/measurement1_151122/temp_test1_151122.obc',
                                 'data/temp_tests/measurement3_151122/temp_test3_151122.obc',
                                np.arange(120, 130))
    ref = np.loadtxt('data/temp_tests/measurement1_151122/temp_test1_151122.obc')
    unal_100 = np.loadtxt('data/temp_tests/measurement3_151122/temp_test3_151122.obc')

    fig, ax = plt.subplots()
    ax.scatter(al_100[:n, 0], al_100[:n, 2], color='b')
    ax.scatter(unal_100[:n, 1], unal_100[:, 3], color='r')
    ax.scatter(ref[:n, 1], ref[:n, 3])
    plt.show()

# %%
