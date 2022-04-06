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


def change_basis(x_new, y_new):
    
    # normalise new unit vectors
    unit_y = y_new/np.sqrt(np.sum(y_new**2))
    unit_x = x_new/np.sqrt(np.sum(x_new**2))
    
    # find x unit vector normal to y and z
    unit_z = np.cross(unit_x, unit_y)

    A = np.vstack((unit_x, unit_y, unit_z))

    return A






if __name__ == '__main__':
    
    x = np.array([1,-1,0])
    y = np.array([1,1,0])

    A = change_basis(x,y)
    print(A)
# %%
