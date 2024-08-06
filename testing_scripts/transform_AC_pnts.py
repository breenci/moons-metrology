import numpy as np
import matplotlib.pyplot as plt
from src.transform.acq_cam import pix2plt
import pandas as pd
from src.transform.plate import polar2cart


def convert_to_polar(Yfiber, Zfiber, roc=4101.1):
    # convert to polar coordinates, it assumes origin (0,0)
    flat_R = np.sqrt(Yfiber**2 + Zfiber**2)
    theta = np.pi/2 - np.arctan2(Zfiber,Yfiber)
    # use positive angles MOONS-1410
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    # convert radial distance into the radial distance 
    # projected onto a curved focal plane
    ratio = abs(flat_R / roc)
    curved_R = roc * np.arcsin(ratio)
    
    return curved_R, theta

if __name__ == '__main__':
    # expand the dimensions of the source position
    source_pos = np.expand_dims(np.array([751, 422]), axis=0)
    source_pos = np.expand_dims(np.array([695, 498]), axis=0)
    source_pos = np.expand_dims(np.array([755, 380]), axis=0)
    
    source_pos[:,0] = source_pos[:,0] - 4
    source_pos[:,1] = source_pos[:,1] + 10
    
    # load the pixel data
    pix_data = np.loadtxt('data/PAE/PAE_04_08/PAE_04_08_01/PAE_04_08_01_ACIM/MOONS2_TCCDACQ18_2024-06-28T09_40_15.txt')
    # pix_data = np.loadtxt('data/PAE/PAE_04_08/PAE_04_08_01/PAE_04_08_01_ACIM/MOONS2_TCCDACQ6_2024-06-28T09_37_49.txt')
    pix_yz = pix_data[:, 0:2]
    
    # load the metrology mask data for comparison
    met_data = np.loadtxt('data/PAE/PAE_04_08/PAE_04_08_01/PAE_04_08_01_METMASK/PAE_04_08_01_METMASK_13.txt')
    
    # load the transformation matrix
    # AC_ID = 1
    t_mat_fn = f'data/PAE/PAE_04_08/PAE_04_08_01/PAE_04_08_01_TMATS/t_mat_13_280624.txt'
    t_mat = np.loadtxt(t_mat_fn)
    
    # tranform to metrology coordinates
    pix_in_plate = pix2plt(pix_yz, t_mat)
    source_in_plate = pix2plt(source_pos, t_mat)
    
    # plot the data in metrology coordinates
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(met_data[:, 2], met_data[:, 3], c='C0', label='Metrology Mask',
               s=100)
    ax.scatter(pix_in_plate[:, 1], pix_in_plate[:, 2], c='C1', marker='x', 
               label='Transformed Centroids', s=100)
    ax.scatter(source_in_plate[0, 1], source_in_plate[0, 2], c='C2', marker='*', 
               label='Transformed Target', s=100)
    ax.set_xlim(np.min(met_data[:, 2])-5, np.max(met_data[:, 2])+5)
    ax.set_ylim(np.min(met_data[:, 3])-5, np.max(met_data[:, 3])+5)
    ax.set_xlabel('Y [mm]')
    ax.set_ylabel('Z [mm]')   
    ax.set_aspect('equal')
    ax.set_title('Metrology Coordinates')
    ax.grid()
    ax.legend()
    
    #  plot the data in pixel coordinates
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pix_yz[:, 0], pix_yz[:, 1], c='C1', marker='x', 
               label='Centroids', s=100)
    ax.scatter(source_pos[0, 0], source_pos[0, 1], c='C2', marker='*',
                label='Target', s=100)
    ax.set_xlim(np.min(pix_yz[:, 0])-50, np.max(pix_yz[:, 0])+50)
    ax.set_ylim(np.min(pix_yz[:, 1])-50, np.max(pix_yz[:, 1])+50)
    ax.set_xlabel('Y [pix]')
    ax.set_ylabel('Z [pix]')
    ax.set_aspect('equal')
    ax.grid()
    ax.set_title('Pixel Coordinates')
    ax.legend()
    
    # convert to polar coordinates
    R_source, theta_source = convert_to_polar(source_in_plate[0][1], source_in_plate[0][2])
    R_met, theta_met = convert_to_polar(met_data[:, 2], met_data[:, 3])
    
    # load the nominal coordinates
    cntrs_fn = 'data/FPU_calibrations/FPUCAL_MAY24/FPU_ARM_LENGTH_DATA_2024-05-28T231144/arm_length_config_full_posbetaV2.csv'
    cntrs = pd.read_csv(cntrs_fn)
    # find the nearest FPU centrea to the source position
    alpha_x, alpha_y, alpha_z = polar2cart(cntrs['alpha_r'], np.deg2rad(cntrs['alpha_theta']), 4101.1)
    source_dist = np.sqrt((alpha_y - source_in_plate[0][1])**2 + (alpha_z - source_in_plate[0][2])**2)
    nearest_5 = np.argsort(source_dist)[:5]
    print(source_dist[nearest_5])
    
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})
    ax.scatter(theta_met, R_met, c='C0', label='Metrology Mask', s=5)
    ax.scatter(theta_source, R_source, c='C2', marker='*', label='Transformed Target', s=100)
    ax.scatter(np.deg2rad(cntrs['alpha_theta']), cntrs['alpha_r'], c='C1', marker='x', label='FPU_centres', s=10)
    ax.scatter(np.deg2rad(cntrs['alpha_theta'][nearest_5]), cntrs['alpha_r'][nearest_5], c='C3', marker='x', label='Nearest 5 FPU centres', s=20)
    ax.legend()
    ax.set_title('Plate Coordinates')
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter( met_data[:, 2], met_data[:, 3], c='C0', label='Metrology Mask',
               s=10)
    ax.scatter(pix_in_plate[:,1], pix_in_plate[:, 2], c='C1', marker='x', 
               label='Transformed Centroids', s=10)
    ax.scatter(alpha_y, alpha_z, c='C3', marker='x', label='FPU_centres', s=10)
    ax.scatter(source_in_plate[0][1], source_in_plate[0][2], c='C2', marker='*', 
               label='Transformed Target', s=10)

    # convert pixel value back to plate coordinates
    # source_in_pix = plt2pix(source_in_plate, t_mat)
    # print('Source position in pixel coordinates:')
    # print(source_in_pix)
    
    # print the important data
    print('Source position in metrology coordinates:')
    print(source_in_plate)
    print('Source position in plate coordinates:')
    print(R_source, np.rad2deg(theta_source))
    print('Nearest 5 FPU centres:')
    print(cntrs['fpu_id'][nearest_5].values)
    
    plt.show()




