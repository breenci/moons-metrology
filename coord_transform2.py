# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plane_fitter(point_coords):
    '''Fit a plane to a set of points and return the unit normal.'''

    # Subtract the centroid from the set of points
    centroid = np.mean(point_coords, axis=0)
    cntrd_pnts = point_coords - centroid
    
    # Perform singular value decomposition
    svd, _, _ = np.linalg.svd(cntrd_pnts.T)
    
    # Final column of the svd gives the unit normal
    norm = svd[:,2]
    
    # Take the +ve norm
    if norm[1]<0:
        norm = -1*norm
    
    return norm, centroid


def change_basis(v1, v2, mode='xy'):
    
    # normalise new unit vectors
    unit2 = v2/np.sqrt(np.sum(v2**2))
    unit1 = v1/np.sqrt(np.sum(v1**2))

    if mode == 'xy':
        unit_x = unit1
        unit_y = unit2
        unit_z = np.cross(unit_x, unit_y)

    elif mode == 'xz':
        unit_x = unit1
        unit_z = unit2
        unit_y = np.cross(unit_z, unit_x)

    elif mode == 'yz':
        unit_y = unit1
        unit_z = unit2
        unit_x = np.cross(unit_y, unit_z)

    A = np.vstack((unit_x, unit_y, unit_z)).T

    return A


def get_translate_mat(new_origin):

    t_mat = np.eye(4)
    t_mat[:3,3] = -1*new_origin

    return t_mat


def get_pln_t_mat(pln_pnts, new_origin, y0):
    
    new_x,_ = plane_fitter(pln_pnts)

    y_dir = y0 - new_origin

    new_z = np.cross(new_x, y_dir)

    cob_mat = change_basis(new_x, new_z, mode='xz')

    return cob_mat


if __name__ == '__main__':

    ten_deg = np.loadtxt('caltest_25_03_22/10_deg.txt')
    coords = ten_deg[:,1:4]

    pln_inds = np.arange(1,17)
    orgn_ind = 6
    y0_ind = 7

    trans_mat = get_pln_t_mat(coords[pln_inds], coords[orgn_ind], coords[y0_ind])

    origin = coords[orgn_ind]

    trnsltd = coords - origin

    transformed = np.matmul(trans_mat.T, trnsltd.T).T

    transformed = transformed[pln_inds]

    plt.scatter(transformed[:,2], transformed[:,0])
    plt.show()








    
    
# %%
