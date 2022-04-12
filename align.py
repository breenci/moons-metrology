# %%
import numpy as np
import matplotlib.pyplot as plt
from get_corr import get_corr
from coord_transform2 import plane_fitter, matrix_transform


def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    A_cntr = np.mean(A, axis=0)
    B_cntr = np.mean(B, axis=0)
    A_var = np.mean(np.linalg.norm(A - A_cntr, axis=1) ** 2)

    covar_mat = np.matmul((A - A_cntr).T, (B - B_cntr)) / n
    U, D, VT = np.linalg.svd(covar_mat)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = np.matmul(np.matmul(U,S), VT)
    c = A_var / np.trace(np.diag(D) @ S)
    t = A_cntr - np.matmul(c * R, B_cntr)

    trans_mat = np.eye(4)
    trans_mat[:3,:3] = c * R
    trans_mat[:3,3] = t 

    return trans_mat



if __name__ == '__main__':
    # load datasets
    cam_mask = np.loadtxt('mask_test/mask_AC_01.txt')
    met_mask = np.loadtxt('mask_test/transformed_coords_mask1_15092021.txt')
    met_mask[:,0] = 0

    # pad and reflect AC camera image in y axis
    cmask_pad = np.hstack((cam_mask, np.zeros((len(cam_mask[:,1]), 1))))
    cmask_pad[:,1] = -1*cmask_pad[:,1]

    a,b = get_corr(met_mask, cmask_pad, (25,11), (13,0))

    srtd_cmask = cmask_pad[b]
    srtd_mmask = met_mask[a]

    norm,_ = plane_fitter(met_mask)

    t_mat = kabsch_umeyama(srtd_mmask, srtd_cmask)


    trans_cmask = matrix_transform(srtd_cmask, t_mat)

    diff = np.abs(srtd_mmask - trans_cmask)
    rms = np.sqrt(np.mean(diff**2, axis=0))
    fig1, ax = plt.subplots()
    ax.scatter(np.arange(len(diff[:,1])), diff[:,2])
    ax.axhline(rms[2])
    plt.show()


# %%
