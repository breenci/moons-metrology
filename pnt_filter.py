
# %%
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
    sizes_out = sizes[in_bnds_mask]

    return in_bnds_mask, data_out, sizes_out


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


if __name__ == '__main__':

    # demonstration of the spherical filter on a test point cloud file
    # load a test file
    unfltrd = np.loadtxt('caltest_22_06_21/transformed/uncalibrated/0_deg_t.txt')

    # do the sphere filtering. Radius of curvature of the plate is 4101.1mm, 
    # centre is assumed to be directly above the zero point => c = (4101.1,0,0)
    # tolerance = 4
    fltrd = sphere_filter_pntcld('caltest_22_06_21/transformed/uncalibrated/0_deg_t.txt',
                                 4101.1, np.array([4101.1, 0, 0]), 4)

    # filter removes any points not on the spherical focal plane. Only fpu dots
    # remain
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(unfltrd[:, 2], unfltrd[:, 3], s=1)
    ax1.set_xlabel('Y (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title('Unfiltered')
    ax2.scatter(fltrd[:, 2], fltrd[:, 3], s=1)
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('Filtered')
    plt.tight_layout()
    plt.show()


# %%
