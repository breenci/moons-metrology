"""
Module to filter spurious points from a metrology ouput point cloud
"""

import numpy as np
import matplotlib.pyplot as plt
from src.align import align_measurements, align_from_file


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


def preprocess_pntcld(fn, template_fn, r=4101.1, c=[4101.1,0,0], rtol=1, sizes=[1.15,1.9,2.6], 
                      stol=.25, m=.0005, template_ids=np.arange(1, 200)):
    """Prepares point cloud for fpu identification"""
    
    # make input lists arrays 
    c_arr = np.array(c)
    size_arr = np.array(sizes)
    
    # load the point cloud file
    pnt_cld = np.loadtxt(fn)
    
    # align to template
    template_data = np.loadtxt(template_fn)
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
    

def points_in_box(coordinates, box_point, box_lengths):
    """
    Find indices of points within a 3D box defined by a point and lengths.

    Parameters:
    - coordinates: List of 3D coordinates (list of lists or numpy array).
    - box_point: Reference point of the box (list or tuple of three values).
    - box_lengths: Lengths of the box along each dimension (list or tuple of three values).

    Returns:
    - List of indices of points within the specified 3D box.
    """
    indices = []

    for i, coord in enumerate(coordinates):
        x_in_range = box_point[0] <= coord[0] <= box_point[0] + box_lengths[0]
        y_in_range = box_point[1] <= coord[1] <= box_point[1] + box_lengths[1]
        z_in_range = box_point[2] <= coord[2] <= box_point[2] + box_lengths[2]

        if x_in_range and y_in_range and z_in_range:
            indices.append(i)

    return indices
    

if __name__ == '__main__':

    # # demonstration of the spherical filter on a test point cloud file
    # # load a test file
    # unfltrd = np.loadtxt('test_filter.txt')

    # # do the sphere filtering. Radius of curvature of the plate is 4101.1mm, 
    # # centre is assumed to be directly above the zero point => c = (4101.1,0,0)
    # # tolerance = 4
    # fltrd = sphere_filter_pntcld('test_filter.txt',
    #                              4101.1, np.array([4101.1, 0, 0]), 4)

    # # filter removes any points not on the spherical focal plane. Only fpu dots
    # # remain
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.scatter(unfltrd[:, 2], unfltrd[:, 3], s=1)
    # ax1.set_xlabel('Y (mm)')
    # ax1.set_ylabel('Z (mm)')
    # ax1.set_title('Unfiltered')
    # ax2.scatter(fltrd[:, 2], fltrd[:, 3], s=1)
    # ax2.set_xlabel('Y (mm)')
    # ax2.set_ylabel('Z (mm)')
    # ax2.set_title('Filtered')
    # plt.tight_layout()


    # ffltrd = make_filter_output('test_filter.txt', c=[4101.4,0,0])

    # fig2, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.scatter(unfltrd[:, 2], unfltrd[:, 3], s=1)
    # ax1.set_xlabel('Y (mm)')
    # ax1.set_ylabel('Z (mm)')
    # ax1.set_title('Unfiltered')
    # ax2.scatter(ffltrd[:, 2], ffltrd[:, 3], s=1)
    # ax2.set_xlabel('Y (mm)')
    # ax2.set_ylabel('Z (mm)')
    # ax2.set_title('Filtered')
    # plt.tight_layout()
    
    coded_fn = 'data/FPM_011223/TEST_03_01/coded_targets_03_01_01.txt'
    pnt_cloud_fn = 'data/FPM_011223/TEST_03_01/FPM_03_01_03.txt'
    
    preprocessed_data = preprocess_pntcld(pnt_cloud_fn, coded_fn, r=4101.1, c=[4101.4, 0, 0], 
                                      stol=0.35)
    
    # plot the preprocessed data
    fig, ax = plt.subplots()
    ax.scatter(preprocessed_data[:, 2], preprocessed_data[:, 3], s=1)
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Preprocessed')
    plt.show()
    
    
    
    