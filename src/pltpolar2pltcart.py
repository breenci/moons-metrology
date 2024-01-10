"""
Module for doing coordinate transformations between cartesian and plate polar
coordinate systems
"""

import numpy as np
import matplotlib.pyplot as plt
from src.do_transform import plt2pix


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
        rho_rad = rho * (180/np.pi)
        theta_rad = theta * (180/np.pi)

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
    # ACQ6
    rho6 = 158.331147  # in mm
    theta6 = 2.335426  # in rad
    plt_roc = 4101.8  # in mm

    # ACQ17
    rho17 = 410.535532
    theta17 = 5.257187

    # ACQ20
    rho20 = 392.96983
    theta20 = 0.688058

    # get x,y,z and plot
    rhos = np.array([rho6, rho17, rho20])
    thetas = np.array([theta6, theta17, theta20])
    labels = np.array(["ACQ6", "ACQ17", "ACQ20"])

    x, y, z = polar2cart(rhos, thetas, plt_roc)

    fig, ax1 = plt.subplots()
    for i in range(len(x)):
        ax1.scatter(y[i], z[i], label=labels[i])
    ax1.scatter(0, 0, label="Origin")
    ax1.set_ylabel("Y (mm)")
    ax1.set_xlabel("Z (mm)")
    plt.legend()
    plt.show()

    pos6 = np.expand_dims(np.array([x[0], y[0], z[0]]), axis=0)
    pix6 = plt2pix(pos6, np.loadtxt("data/mask_test/AC_mats/t_mat_AC6.txt"))

    pos17 = np.expand_dims(np.array([x[1], y[1], z[1]]), axis=0)
    pix17 = plt2pix(pos6, np.loadtxt("data/mask_test/AC_mats/t_mat_AC17.txt"))

    pos20 = np.expand_dims(np.array([x[2], y[2], z[2]]), axis=0)
    pix20 = plt2pix(pos20, np.loadtxt("data/mask_test/AC_mats/t_mat_AC20.txt"))
