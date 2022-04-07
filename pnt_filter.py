
# %%
import numpy as np
import matplotlib.pyplot as plt
from coord_transform2 import do_transform
import glob
import os.path


def sphere_filter(data, r, c, tol):
    """Filters out any points not on a spherical surface to within a tolerance"""

    # extract coords
    coords = data[:,1:4]

    # Subtract centre of sphere from coords to satisfy equation of sphere 
    cntrd_coords =  coords - c

    # Calculate how far each point is from sphere surface
    dist = np.sqrt(np.sum(cntrd_coords**2, axis=1))
    diff = abs(dist - r)

    # Reject points not on surface to within tol
    out_coords = coords[diff < tol]

    return out_coords


def size_filter(data, size_arr, tol):
    """Filters out points not within a tolerance of given sizes"""

    # Extract sizes
    sizes = data[:, 7]

    # Define upper and lower bounds
    upper_bnds = size_arr + tol
    lower_bnds = size_arr - tol

    # Create logical mask for first set of bounds then loop through the other
    # bounds and update the mask with each loop
    in_bnds_mask = (sizes < upper_bnds[0]) & (sizes > lower_bnds[0])

    for i in range(1, len(upper_bnds)):
        cond_mask = (sizes < upper_bnds[i]) & (sizes > lower_bnds[i])
        in_bnds_mask = in_bnds_mask | cond_mask

    # Filter dataset using mask
    data_out = data[in_bnds_mask]

    return data_out


if __name__ == '__main__':
    ten_deg = np.loadtxt('caltest_25_03_22/10_deg.txt')
    
    coords = ten_deg[:,1:4]

    pln_inds = np.arange(1,17)
    orgn_ind = 293
    y0_ind = 7

    ten_deg[:,1:4] = do_transform(coords, pln_inds, orgn_ind, y0_ind)

    fltrd = sphere_filter(ten_deg, 4101.4, (4101.4,0,0), 4)
    sfiltrd = size_filter(ten_deg, np.array([2.6,1.8,1.1]), .4)

    plt.scatter(fltrd[:,1], fltrd[:,2], s=1)
    plt.show()


# %%
