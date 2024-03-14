"""
Module to filter spurious points from a metrology ouput point cloud
"""

import numpy as np
import matplotlib.pyplot as plt


def sphere_filter(coords, r, c, tol):
    """
    Filters out any points not on a spherical surface to within a tolerance

    :param coords: 3D point cloud coordinates with X, Y, and Z in columns
    :type data: N x 3 numpy array
    :param r: Radius of sphere which defines filtering surface
    :type r: float
    :param c: Centre of sphere which defines filtering surface
    :type c: 1 x 3 numpy array
    :param tol: values within this tolerance are accepted
    :type tol: float
    :return: Filtered 3D point cloud coordinates with X, Y, and Z in columns
    :rtype: N x 3 numpy array
    """

    # Subtract centre of sphere from coords to satisfy equation of sphere
    cntrd_coords = coords - c

    # Calculate how far each point is from sphere surface
    dist = np.sqrt(np.sum(cntrd_coords**2, axis=1))
    diff = abs(dist - r)

    # Reject points not on surface to within tol
    sphere_mask = diff < tol
    out_coords = coords[sphere_mask]

    return sphere_mask, out_coords


def size_filter(coords, sizes, size_arr, tol):
    """
    Filters out points not within a tolerance of given sizes

    :param coords: 3D point cloud coordinates with X, Y, and Z in columns
    :type data: N x 3 numpy array
    :param sizes: Array of point sizes
    :type sizes: 1d numpy array
    :param size_arr: Array of desired sizes
    :type size_arr: 1d numpy array
    :param tol: tolerance
    :type tol: float
    :return: Filtered data
    :rtype: N x 3 numpy array
    """

    # Define upper and lower bounds
    upper_bnds = size_arr + tol
    lower_bnds = size_arr - tol
    
    # raise an error is bnds overlap
    diff = np.diff(size_arr)
    if np.any(diff < 2*tol):
        raise ValueError('Bounds overlap')
    
    # Create logical mask for first set of bounds then loop through the other
    # bounds and update the mask with each loop
    in_bnds_mask = (sizes < upper_bnds[0]) & (sizes > lower_bnds[0])
    # Make sizes discrete and match to known values
    sizes[in_bnds_mask] = size_arr[0]

    for i in range(1, len(upper_bnds)):
        cond_mask = (sizes < upper_bnds[i]) & (sizes > lower_bnds[i])
        sizes[cond_mask] = size_arr[i]
        in_bnds_mask = in_bnds_mask | cond_mask

    # Filter dataset using mask
    data_out = coords[in_bnds_mask]
    sizes_class = sizes[in_bnds_mask]

    return in_bnds_mask, data_out, sizes_class


def lin_corr(coords, sizes, m):
    """Function to filter points based on a linear size relationship"""
    
    D_rad = np.sqrt(np.sum(coords**2, axis=1))
    
    # Calculate the corrected size based on the linear relationship
    corr_sizes = sizes - m*D_rad
    
    return corr_sizes


def sphere_filter_pntcld(fn, r, c, tol):
    """Apply sphere filter to a moons point cloud file"""

    # load the point cloud file
    pntcld_data = np.loadtxt(fn)

    # extract coordinates
    coords = pntcld_data[:, 1:4]

    # get the mask which will filter unwanted values and apply to coords
    sphere_mask, _ = sphere_filter(coords, r, c, tol)
    pntcld_fltrd = pntcld_data[sphere_mask]

    return pntcld_fltrd


def size_filter_pntcld(fn, size_arr, tol):
    """Apply size filter to a moons point cloud file"""

    # load the point cloud file
    pntcld_data = np.loadtxt(fn)

    # extract coords and point sizes
    coords = pntcld_data[:, 1:4]
    sizes = pntcld_data[:, 7]

    # get the mask which will filter unwanted values and apply to coords
    size_mask, _ = size_filter(coords, sizes, size_arr, tol)
    pntcld_fltrd = pntcld_data[size_mask]

    return pntcld_fltrd


def make_filter_output(fn, r=4101.1, c=[4101.1,0,0], rtol=4, sizes=[1.15,1.9,2.6], stol=.25):
    """Prepares point cloud for fpu identification"""
    
    # make input lists arrays 
    c_arr = np.array(c)
    size_arr = np.array(sizes)

    # do sphere filtering
    sp_fltrd_pc = sphere_filter_pntcld(fn, r, c_arr, rtol)

    # do size filtering
    size_mask,_,new_sizes = size_filter(sp_fltrd_pc[:,1:4], sp_fltrd_pc[:,7], size_arr, stol)
    ffltrd = sp_fltrd_pc[size_mask]

    # make sizes discrete
    ffltrd[:, 7] = new_sizes

    # flip to left handed coordinate system
    ffltrd[:,1] = -1*ffltrd[:,1]

    return ffltrd


def matrix_transform(coords, trans_mat):
    """Do the matrix transformation"""

    # pad input coords with ones to allow translation
    pad_coords = np.vstack((coords.T, np.ones_like(coords[:,0])))

    # do transformation
    pad_trans_coords = np.matmul(trans_mat, pad_coords)

    # unpad
    trans_coords = pad_trans_coords[:3,:].T

    return trans_coords


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


def preprocess_pntcld(fn, align_ref_fn, r=4101.1, c=[4101.1,0,0], rtol=1, sizes=[1.15,1.9,2.6], 
                      stol=.35, m=.0005, template_ids=np.arange(1, 200)):
    """Prepares point cloud for fpu identification"""
    
    # make input lists arrays 
    c_arr = np.array(c)
    size_arr = np.array(sizes)
    
    # load the point cloud file
    pnt_cld = np.loadtxt(fn)
    
    # align to template
    template_data = np.loadtxt(align_ref_fn)
    al_ffltrd = align_measurements(template_data, pnt_cld, template_ids)

    # do sphere filtering
    sp_mask, sp_fltrd = sphere_filter(al_ffltrd, r, c_arr, rtol)
    
    sp_fltrd_pc = np.copy(pnt_cld)
    sp_fltrd_pc = sp_fltrd_pc[sp_mask] 
    sp_fltrd_pc[:,1:4] = sp_fltrd
    
    # correct sizes
    sp_fltrd_pc[:,7] = lin_corr(sp_fltrd_pc[:,1:4], sp_fltrd_pc[:,7], m)
    
    # do size filtering
    size_mask,_,new_sizes = size_filter(sp_fltrd_pc[:,1:4], sp_fltrd_pc[:,7], size_arr, stol)
    ffltrd = sp_fltrd_pc[size_mask]
    ffltrd[:, 7] = new_sizes
    
    # flip to left handed coordinate system
    ffltrd[:,1] = -1*ffltrd[:,1]
    
    return ffltrd
    

if __name__ == '__main__':

    coded_fn = 'data/FULLPLATE_250923/FULLPLATE_01_01_01_TEM/FULLPLATE_01_01_01_TEM_HIN.txt'
    pnt_cloud_fn = 'data/FPM_011223/TEST_03_01/FPM_03_01_03.txt'
    
    preprocessed_data = preprocess_pntcld(pnt_cloud_fn, coded_fn, r=4101.1, c=[4101.4, 0, 0], 
                                      stol=0.35)
    
    # plot the raw data
    raw_data = np.loadtxt(pnt_cloud_fn)
    template_data = np.loadtxt(coded_fn)
    fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
    ax[0].scatter(raw_data[:, 2], raw_data[:, 3], s=1)
    ax[0].set_ylabel('Z (mm)')
    ax[0].set_xlabel('Y (mm)')
    ax[1].scatter(preprocessed_data[:, 2], preprocessed_data[:, 3], s=1)
    ax[1].scatter(template_data[:, 2], template_data[:, 3], s=1)
    ax[1].set_xlabel('Y (mm)')
    ax[1].set_title('Preprocessed')
    plt.show()