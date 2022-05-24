# %%
import numpy as np
import matplotlib.pyplot as plt
from get_corr import get_corr
from coord_transform2 import matrix_transform


def kabsch_umeyama(A, B):
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
    R = np.matmul(np.matmul(U,S), VT)
    c = A_var / np.trace(np.matmul(np.diag(D), S))
    t = A_cntr - np.matmul(c * R, B_cntr)

    # bring all together to create augmented transformation matrix
    trans_mat = np.eye(4)
    trans_mat[:3,:3] = c * R
    trans_mat[:3,3] = t 

    return trans_mat


def align_measurements(fns, targ_ids):

    # load ref id and xyz
    ref = np.loadtxt(fns[0], usecols=(0,1,2,3))
    # get targets from input ids
    ref_targs = ref[np.searchsorted(ref[:,0], targ_ids), 1:4]
    plt.scatter(ref_targs[:,0], ref_targs[:,1])

    # load each unaligned data set and align to ref using KU
    for n in range(len(fns)-1):
        unaligned = np.loadtxt(fns[n+1], usecols=(0,1,2,3))
        una_targs = unaligned[np.searchsorted(unaligned[:,0], targ_ids), 1:4]
        plt.scatter(una_targs[:,0], una_targs[:,1])
        # align to ref
        aligned = matrix_transform(unaligned[:,1:4], kabsch_umeyama(ref_targs, una_targs))

    return aligned


if __name__ == '__main__':
    # # load datasets
    # cam_mask = np.loadtxt('mask_test/mask_AC_01.txt')
    # met_mask = np.loadtxt('mask_test/transformed_coords_mask1_15092021.txt')
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

    
    al_100 = align_measurements(['caltest_25_03_22/10_deg.txt', 'caltest_25_03_22/100_deg.txt'], np.arange(50,61))
    ref = np.loadtxt('caltest_25_03_22/10_deg.txt')
    unal_100 = np.loadtxt('caltest_25_03_22/100_deg.txt')

    fig, ax = plt.subplots()
    ax.scatter(al_100[:,0], al_100[:,2])
    ax.scatter(unal_100[:,1], unal_100[:,3], color='r')
    ax.scatter(ref[:,1], ref[:,3])
    plt.show()

# %%
