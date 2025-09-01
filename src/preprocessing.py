"""
Module to filter spurious points from a metrology ouput point cloud. 

This module contains functions to filter points based on their distance from a
spherical surface and their size. It also includes functions to align point
clouds using the Kabsch-Umeyama algorithm, which is useful for aligning
measurements to a reference point cloud. The module provides functionality for
preprocessing the raw output of the Hexagon Metrology System to prepare it
for input into the FPU identification algorithm.

Author: Ciarán Breen (UK ATC)
"""

import numpy as np
from src.pnt_filter import sphere_filter, size_filter
from src.align import align_measurements


def lin_corr(coords, sizes, m):
    """Correct sizes as a linear function of radial distance from origin
    
    :param coords: 3D point cloud coordinates with X, Y, and Z in columns
    :type coords: N x 3 numpy array
    :param sizes: Array of point sizes
    :type sizes: 1d numpy array
    :param m: slope of the linear relationship
    :type m: float
    :return: Corrected sizes
    :rtype: 1d numpy array"""
    
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


def full_filter(pnt_cld, r=4101.1, c=[4101.1,0,0], rtol=1, sizes=[1.15,1.9,2.6], 
                      stol=.35, m=.0005):
    
    # make input lists arrays 
    c_arr = np.array(c)
    size_arr = np.array(sizes)

    coords = pnt_cld[:, 1:4]
    # do sphere filtering
    sp_mask, sp_fltrd = sphere_filter(coords, r, c_arr, rtol)
    
    sp_fltrd_pc = np.copy(pnt_cld)
    sp_fltrd_pc = sp_fltrd_pc[sp_mask] 
    sp_fltrd_pc[:,1:4] = sp_fltrd
    
    # correct sizes
    sp_fltrd_pc[:,7] = lin_corr(sp_fltrd_pc[:,1:4], sp_fltrd_pc[:,7], m)
    
    # do size filtering
    size_mask,_,new_sizes = size_filter(sp_fltrd_pc[:,1:4], sp_fltrd_pc[:,7], size_arr, stol)
    ffltrd = sp_fltrd_pc[size_mask]
    ffltrd[:, 7] = new_sizes
    
    return ffltrd
    
    
def preprocess_pntcld(fn, align_ref_fn, r=4101.1, c=[4101.1,0,0], rtol=1, sizes=[1.15,1.9,2.6], 
                      stol=.35, m=.0005, template_ids=np.arange(1, 200), save_aligned=False, save_fn=None):
    """Prepares point cloud for fpu identification
    
    Loads a point cloud file, aligns it to a template using the Kabsch-Umeyama,
    applies a sphere filter, and then applies a size filter. Flips the
    coordinates to a left-handed coordinate system to match the plate coordinate
    system defined in VLT-TRE-MON-14620-3007 RFE Software Design Description.
    
    Aligned, unfiltered point cloud can be optionaly saved to a file.
    
    :param fn: Path to the point cloud file
    :type fn: str
    :param align_ref_fn: Path to the template point cloud file for alignment
    :type align_ref_fn: str
    :param r: Radius of sphere which defines filtering surface, defaults to 4101.1
    :type r: float, optional
    :param c: Centre of sphere which defines filtering surface, defaults to [4101.1, 0, 0]
    :type c: list, optional
    :param rtol: Tolerance for sphere filtering, defaults to 1
    :type rtol: float, optional
    :param sizes: Array of desired sizes for size filtering, defaults to [1.15, 1.9, 2.6]
    :type sizes: list, optional
    :param stol: Tolerance for size filtering, defaults to 0.35
    :type stol: float, optional
    :param m: Slope of the linear relationship for size correction, defaults to 0.0005
    :type m: float, optional
    :param template_ids: IDs of points to be used in alignment, defaults to np.arange(1, 200)
    :type template_ids: numpy.ndarray, optional
    :param save_aligned: Whether to save the aligned point cloud, defaults to False
    :type save_aligned: bool, optional
    :param save_fn: Path to save the aligned point cloud if save_aligned is True
    :type save_fn: str, optional
    :return: Filtered and aligned point cloud data
    :rtype: numpy.ndarray"""
    
    # make input lists arrays 
    c_arr = np.array(c)
    size_arr = np.array(sizes)
    
    # load the point cloud file
    pnt_cld = np.loadtxt(fn)
    
    # align to template
    template_data = np.loadtxt(align_ref_fn)
    al_ffltrd = align_measurements(template_data, pnt_cld, template_ids)
    
    if save_aligned:
        if save_fn is None:
            raise ValueError('save_fn must be specified if save_aligned is True')
        al_pntcld = np.copy(pnt_cld)
        al_pntcld[:, 1:4] = al_ffltrd
        # flip to left handed coordinate system
        al_pntcld[:,1] = -1*al_pntcld[:,1]
        np.savetxt(save_fn, al_pntcld, fmt='%s')


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