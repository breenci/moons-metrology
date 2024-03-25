"""
Module for doing coordinate transformations between cartesian and plate polar
coordinate systems
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.do_transform import plt2pix
from src.calibrations import read_FPU_out


def polar2cart(rho, theta, roc, angle_unit='rad'):
    """Convert from plate polar to plate cartesian coordinate systems

    :param rho: Radial coordinate of target
    :type rho: float
    :param theta: Angular coordinate of target
    :type theta: float
    :param roc: Radius of curvature of plate
    :type roc: float
    :param angle_unit: unit for input angles. Radians or degrees, defaults to 'rad'
    :type angle_unit: str, optional
    :return: Plate cartesian coordinates of target
    :rtype: tuple
    """

    # covert to radians if necessary
    if angle_unit == 'deg':
        rho_rad = rho
        theta_rad = np.deg2rad(theta)

    elif angle_unit == 'rad':
        rho_rad = rho
        theta_rad = theta

    # define angle phi between centre of curvature and target
    phi = rho_rad/roc

    # define projection of r onto flat plane
    r_proj = roc * np.sin(phi)

    # return x,y,z
    x_plt = roc * (np.cos(phi) - 1)
    z_plt = r_proj * np.cos(theta_rad)
    y_plt = r_proj * np.sin(theta_rad)

    return x_plt, y_plt, z_plt


if __name__ == '__main__':
    # load in the data
    new_cntrs_fn = 'data/FPU_calibrations/FPUCAL_03/FPU_ALPHA_CENTRES/results20240325-092858.csv'
    old_met_cntrs_fn = 'data/FULLPLATE_250923/FULLPLATE_10/Positioners-Centers-260224.txt'
    
    new_cntrs = pd.read_csv(new_cntrs_fn)
    old_cntrs = np.loadtxt(old_met_cntrs_fn)
    
    # convert old centres to cartesian
    new_x, new_y, new_z = polar2cart(new_cntrs['r_centre'], new_cntrs['theta_centre'], 4101.1, angle_unit='deg')
    
    # plot the new and old centres
    fig, ax = plt.subplots()
    ax.scatter(new_y, new_z, label='New Centres')
    ax.scatter(old_cntrs[:,6], old_cntrs[:,7], label='Old Centres')
    ax.scatter(new_y[0], new_z[0], label='FPU 12', color='red')
    ax.scatter(old_cntrs[0,6], old_cntrs[0,7], label='FPU 12', color='red')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    plt.show()
    
    
    
    
