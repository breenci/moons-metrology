# %%
from ast import Raise
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




    
    
# %%
