# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.neighbors import NearestNeighbors
from coord_transform import transform_data

plt.close('all')

def read_fits(fname):
    '''Reads the fits file and return xy values'''
    
    hdulist = fits.open(fname)
    record = hdulist[1].data
    
    x_vals = record.xpix     
    y_vals = record.ypix

    out_xy = np.column_stack((x_vals, y_vals))
    
    return out_xy


def read_obc(fname):
    metr_data = pd.read_fwf(fname, usecols=range(1,8))
    metr_coords = metr_data.iloc[:, 1:4].to_numpy()

    return metr_coords


def rescale(mask1, mask2, ref1, ref2):
    """Rescale mask2 so that the distance between two reference points is
    the same in mask1 and mask2.

    :param mask1: array containing mask coords at reference scale
    :type mask1: N X 3 numpy array
    :param mask2: array containg mask coords to be rescaled
    :type mask2: N X 3 numpy array
    :param ref1: indices of reference points in mask1
    :type ref1: tuple
    :param ref2: indices of reference points in mask2
    :type ref2: tuple
    :return: rescaled mask2 coordinates
    :rtype: N X 3 numpy array
    """

    # calculate ref distance in both coordinate systems
    mask1_dist = dist_calc(mask1[ref1[0], :], mask1[ref1[1], :])
    mask2_dist = dist_calc(mask2[ref2[0], :], mask2[ref2[1], :])

    # Apply scaling factor and return rescaled mask2 data
    scale = mask1_dist/mask2_dist
    rescaled_data = scale*mask2
    
    return rescaled_data


def dist_calc(pnt1,pnt2):
    """Calculate the distance between 2 points"""    

    # convert points to numpy arrays
    pnt1_arr = np.array(pnt1)
    pnt2_arr = np.array(pnt2)

    dist =  np.sqrt(np.sum((pnt1_arr - pnt2_arr)**2))

    return dist


def match_near_nbrs(data1, data2, tol):
    """Use nearest neighbours analysis to find the indices of corresponding
    points between the two datasets.  

    :param data1: numpy array containg first dataset
    :param data2: numpy array containing second dataset
    :param tol: distance tolerance. data points outside of this tolerance are 
    ignored.
    :type tol: float
    :return: two arrays of corresponding indices i.e. data1[idx1_out] 
    corresponds to data2[idx2_out]
    :rtype: 2 numpy arrays
    """    
    
    # Stack datasets for nearest neighbours analysis
    full_data = np.vstack((data1, data2))
    
    # Nearest neighbours analysis
    nbrs = NearestNeighbors(n_neighbors=1, 
                            algorithm='ball_tree').fit(full_data)
    distances, indices = nbrs.kneighbors()
    
    # get indices data2
    data1_idx = np.arange(0, len(data1))
    data2_idx = indices[data1_idx] - len(data1)
    
    # reject points where NN is too far away. Indicates a missing point.
    distances1 = distances[data1_idx]
    idx1_out = data1_idx[distances1[:, 0] < tol]
    idx2_out = data2_idx[distances1 < tol]
    
    return idx1_out, idx2_out
        

def get_corr(mask1, mask2, ref1, ref2):
    '''Finds corresponding points between the two mask datasets. Rough aligns
    mask2 to mask1 using common reference points and then uses nearest 
    neighbours analysis to find indices of corresponding points'''
    
    # Transform mask data into coordinate system defined by common reference
    # points to rough align
    mask1_idxs = np.arange(0, len(mask1))
    _, RA_mask1 = transform_data(mask1, ref1[0], mask1_idxs, ref1[1])
    
    mask2_idxs = np.arange(0, len(mask2))    
    _, trans_mask2 = transform_data(mask2, ref2[0], mask2_idxs, ref2[1])
    
    # rescale mask2 to complete rough alignment
    RA_mask2 = rescale(RA_mask1, trans_mask2, ref1, ref2)

    # NN analysis
    ind1, ind2 = match_near_nbrs(RA_mask1, RA_mask2, 0.3)
    

    return ind1, ind2


if __name__ == '__main__':

    # Inputs
    mask = 1
    metr_fn = 'transformed_coords_new_def.obc'

    if mask==2:
        
        metr_origin_idx = 19 # 21
        metr_z0_idx = 15 # 18

        cam_fn = 'association_table_AC_02.fits'
        cam_origin_idx = 13
        cam_z0_idx = 6

        ind = [27, 54]

    if mask==1:
        
        metr_origin_idx = 25
        metr_z0_idx = 11

        cam_fn = 'association_table_AC_01.fits'
        cam_origin_idx = 13
        cam_z0_idx = 0 
        
        ind = [0, 27]


    # Read obc file and extract coords
    metr_coords = read_obc(metr_fn)
        
    # Extract AC mask coordinates
    mask_coords = metr_coords[ind[0]:ind[1]]

    # read cam data and insert z dimension
    cam_coords = read_fits(cam_fn)
    z = np.zeros_like(cam_coords[:, 0])
    cam_coords = np.insert(cam_coords,2,z, axis=1)
    # apply reflection in y dir
    cam_coords[:,1] = -1*cam_coords[:,1]


    # get correspondances between mask datasets 
    m1, m2 = get_corr(mask_coords, cam_coords, (metr_origin_idx, metr_z0_idx), 
                     (cam_origin_idx, cam_z0_idx))


# %%
