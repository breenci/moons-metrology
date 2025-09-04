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


def convert_to_polar(Yfiber, Zfiber, roc=4101.1):
    """Convert fiber coordinates to polar coordinates.

    :param Yfiber: Y coordinate of the fiber
    :type Yfiber: float
    :param Zfiber: Z coordinate of the fiber
    :type Zfiber: float
    :param roc: Radius of curvature of the plate
    :type roc: float
    :return: Curved radial coordinate and angular coordinate
    :rtype: tuple
    """
    # convert to polar coordinates, it assumes origin (0,0)
    flat_R = np.sqrt(Yfiber**2 + Zfiber**2)
    theta = np.pi/2 - np.arctan2(Zfiber,Yfiber)
    
    # use positive angles 
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    # convert radial distance into the radial distance 
    # projected onto a curved focal plane
    ratio = abs(flat_R / roc)
    curved_R = roc * np.arcsin(ratio)
    
    return curved_R, theta