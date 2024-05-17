"""
Code for transforming coordinates between pixel and plate coordinate.

Author: Ciarán Breen
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from src.transform.spatial import matrix_transform


def pix2plt(coords_arr, tmat_arr):
    """Transform from pixel to plate coordinate systems"""
    # pad z
    pad_coords = np.insert(coords_arr, 2, 0, axis=1)

    # do transform
    out_coords = matrix_transform(pad_coords, tmat_arr)

    return out_coords


def plt2pix(coords_arr, mat_arr):
    """Transform from plate to pixel coordinate systems"""

    # take inverse. Note: Assumes input matrix describes pixel to
    # plate transformation
    plt2pix_mat = np.linalg.inv(mat_arr)

    # do transform
    plt_in_pix = matrix_transform(coords_arr, plt2pix_mat)

    # remove z axis
    out_coords = plt_in_pix[:, 0:2]

    return out_coords


def transform_from_file(coords_fn, mat_fn, out_fn=None, mode='pix2plt'):
    """Apply a coordinate transformation where coordinates and transformation
    matrices are in text file format

    :param coords_fn: Name of file containing coordinates to be transformed
    :type coords_fn: Numpy array with xyz values in columns
    :param mat_fn: Name of file containing transformation matrix
    :type mat_fn: 4 x 4 numpy array
    :param out_fn: Path to saved output file. If no path is given the output file is not saved.
    :type out_fn: str, optional
    :param mode: Specifies the direction of transformation, defaults to 'pix2plt'
    :type mode: str, optional
    :return: Transformed coordinates
    :rtype: Numpy array with xyz (pix2plt) or xy (plt2pix) values in columns
    """

    # load files
    coords = np.loadtxt(coords_fn)
    trans_mat = np.loadtxt(mat_fn)

    # choose direction of transformation. Pixel to plate (pix2plt) or
    # plt2pix. Note: This assumes that the input matrix describes the
    # pix2plt mode
    if mode == 'pix2plt':
        out_coords = pix2plt(coords, trans_mat)

    if mode == 'plt2pix':
        out_coords = plt2pix(coords, trans_mat)

    # save output as .txt file
    if out_fn is not None:
        np.savetxt(out_fn, out_coords)

    return out_coords


if __name__ == '__main__':
    
    fig, ax = plt.subplots()
    for mask in np.sort(glob.glob('dummy_trans_mats/*')):
        mask_id = mask[23:25]

        plt_coords = transform_from_file('mask_test/mask_AC_01.txt', mask)

        ax.scatter(plt_coords[:, 0], plt_coords[:, 1], s=1)

    plt.show()