"""
Module for doing coordinate transformations between cartesian and plate polar
coordinate systems
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def polar2cart(rho, theta, roc, angle_unit='rad'):
    """Convert from plate polar to plate cartesian coordinate systems

    :param rho: Radial coordinate of target
    :type rho: float
    :param theta: Angular coordinate of target
    :type theta: float
    :param roc: Radius of curvature of plate
    :type roc: float
    :param angle_unit: unit for input angles. Radians ('rad') or degrees ('deg')
        , defaults to 'rad'
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