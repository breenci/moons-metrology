import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.calibrations import read_metro_out, read_FPU_out


def convert_to_cartesian(r, theta):
    '''
    INPUTS:
    - r     : Radius (float)
    - theta : Angle (float)
    
    OUTPUTS:
    - x     : x-coordinate (float)
    - y     : y-coordinate (float)
    '''

    x = r * np.cos(np.deg2rad(theta))  # convert to radians
    y = r * np.sin(np.deg2rad(theta))  # convert to radians
    
    return (x, y)


def convert_to_polar(x, y, wrap=True):
    '''
    INPUTS:
    - x     : x-coordinate (float)
    - y     : y-coordinate (float)
    
    OUTPUTS:
    - r     : Radius (float)
    - theta : Angle (float)
    '''
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if wrap:
        theta[theta < 0] += 2*np.pi  # convert to 0-360
    return (r, np.rad2deg(theta))


# write a function to rotate the data
def rotate_data(Y, Z, angle):
    '''
    INPUTS:
    - Y     : Y-coordinate (np.array)
    - Z     : Z-coordinate (np.array)
    - angle : Angle to rotate by (float)
    
    OUTPUTS:
    - Y_rot : Rotated Y-coordinate (np.array)
    - Z_rot : Rotated Z-coordinate (np.array)
    '''
    # convert to radians
    angle_rad = np.deg2rad(angle)
    
    # create the rotation matrix
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # rotate the data
    Y_rot, Z_rot = np.dot(R, np.vstack((Y, Z)))
    
    return Y_rot, Z_rot


if __name__ == "__main__":
    # get the data
    inter_data_fns = sorted(glob.glob("data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/**/arm_length_intermediate_data.csv", recursive=True))
    cntrs_fn = "data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/fpu_centres.csv"
    
    cntrs = pd.read_csv(cntrs_fn).sort_values(by='FPU', ignore_index=True)
    cntrs['x_centre'] = -1*cntrs['x_centre']
    
    expected_theta = np.array([-180., -137.625, -95.25, -52.875, -10.5, 31.875, 74.25, 116.625])
    expected_beta = np.array([-172., -132.125, -92.25, -52.375, -12.5, 27.375, 67.25, 107.125])
    alpha0 = 142
    
    # load the data into numpy arrays
    # intialise arrays to store the data
    Z_dict = {}
    Y_dict = {}

    # load the first file to get the fpu ids
    full_df = pd.read_csv(inter_data_fns[0]).sort_values(by='fpu_id', ignore_index=True)
    Z_dict["fpu_ids"] = full_df['fpu_id']
    Y_dict["fpu_ids"] = full_df['fpu_id']
    
    # loop through the files and store the data in a dictionary
    for fn in inter_data_fns:
        # get the alpha position from the file name
        folder_name = fn.split('/')[-2]
        alpha_pos = folder_name.split('_')[2]
        
        # load the data
        full_df = pd.read_csv(fn).sort_values(by='fpu_id', ignore_index=True)
        
        # make sure that the fpu ids are the same in each file
        if not np.array_equal(full_df['fpu_id'], Z_dict['fpu_ids']):
            raise ValueError("FPU IDs are not the same in each file")
        
        # extract the beta centre positions
        Cbeta_df = full_df[['fpu_id', 'beta_r', 'beta_theta']]
        
        # store the data in the arrays
        Z_dict["Z_" + alpha_pos], Y_dict["Y_" + alpha_pos] = convert_to_cartesian(Cbeta_df['beta_r'], Cbeta_df['beta_theta'])
        
         
    # convert the dictionary to a dataframe
    Y_df = pd.DataFrame(Y_dict).set_index('fpu_ids')
    Z_df = pd.DataFrame(Z_dict).set_index('fpu_ids')
    
    
    # subtract the centre positions
    Y_corr = Y_df.to_numpy() - cntrs['y_centre'].to_numpy().reshape(-1,1)
    Z_corr = Z_df.to_numpy() - cntrs['x_centre'].to_numpy().reshape(-1,1)
    
    # rotate each set of YZ data by 142 degrees and cobvert to polar
    Y_rot = np.zeros(Y_corr.shape)
    Z_rot = np.zeros(Z_corr.shape)
    r_rot = np.zeros(Y_rot.shape)
    theta_rot = np.zeros(Z_rot.shape)
    
    for i in range(Y_corr.shape[1]):
        Y_rot[:,i], Z_rot[:,i] = rotate_data(Y_corr[:,i], Z_corr[:,i], alpha0)
        r_rot[:,i], theta_rot[:,i] = convert_to_polar(Y_rot[:,i], Z_rot[:,i], wrap=False)
        
    # wrap the angles to 0-360
    theta_rot[theta_rot < 0] += 360
    expected_theta[expected_theta < 0] += 360
    
    theta_diff = theta_rot - expected_theta
    theta_diff_mean = np.mean(theta_diff, axis=1)
    
    # load the data from alpha 1 folder
    alpha1_fns = sorted(glob.glob("data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/ALPHA_POSITION_1_2024-05-05T160410/*txt"))
    arm_length_fn = 'data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/arm_length_data.csv'
    
    Z_beta_dict = {}
    Y_beta_dict = {}
    
    # read the first file to get the fpu ids
    alpha1_pos = read_FPU_out(alpha1_fns[0])
    fpu_ids = alpha1_pos['ID']
    
    Z_beta_dict['fpu_ids'] = fpu_ids
    Y_beta_dict['fpu_ids'] = fpu_ids
    
    for fn in alpha1_fns:
        folder_name = fn.split('/')[-1]
        beta_pos = folder_name.split('_')[4][:2]
        
        full_beta_df = read_FPU_out(fn)
        beta_df = full_beta_df[['ID', 'R', 'Theta']]
        
        # make sure that the fpu ids are the same in each file
        if not np.array_equal(beta_df['ID'], fpu_ids):
            raise ValueError("FPU IDs are not the same in each file")
        
        # extract the beta centre positions
        # store the data in the arrays
        Z_beta_dict["Z_" + beta_pos], Y_beta_dict["Y_" + beta_pos] = convert_to_cartesian(beta_df['R'], beta_df['Theta'])
        
    # convert the dictionary to a dataframe
    Y_beta_df = pd.DataFrame(Y_beta_dict).set_index('fpu_ids')
    Z_beta_df = pd.DataFrame(Z_beta_dict).set_index('fpu_ids')
    
    # subtract the centre positions
    Y_beta_corr = Y_beta_df.to_numpy() - cntrs['y_centre'].to_numpy().reshape(-1,1)
    Z_beta_corr = Z_beta_df.to_numpy() - cntrs['x_centre'].to_numpy().reshape(-1,1)
    
    # rotate each set of YZ data by 142 degrees and cobvert to polar
    Y_beta_rot = np.zeros(Y_beta_corr.shape)
    Z_beta_rot = np.zeros(Z_beta_corr.shape)
    r_beta_rot = np.zeros(Y_beta_rot.shape)
    theta_beta_rot = np.zeros(Z_beta_rot.shape)
    
    for i in range(Y_beta_corr.shape[1]):
        Y_beta_rot[:,i], Z_beta_rot[:,i] = rotate_data(Y_beta_corr[:,i], Z_beta_corr[:,i], alpha0)
    
    # subtract the beta centre position
    # Y_beta_rot = Y_beta_rot - Y_corr[:,0].reshape(-1,1)
    # Z_beta_rot = Z_beta_rot - Z_corr[:,0].reshape(-1,1)
    
    arm_lengths = pd.read_csv(arm_length_fn).sort_values(by='fpu_id', ignore_index=True)
    L_alpha = arm_lengths['alpha_length']
    
    # Y_beta_rot = Y_beta_rot + L_alpha.to_numpy().reshape(-1,1)
    Y_beta_rot = Y_beta_rot + 8
    
    # convert to polar
    for i in range(Y_beta_rot.shape[1]):
        r_beta_rot[:,i], theta_beta_rot[:,i] = convert_to_polar(Y_beta_rot[:,i], Z_beta_rot[:,i], wrap=False)
    
    theta_beta_rot[theta_beta_rot < 0] += 360
    expected_beta[expected_beta < 0] += 360
    
    print(expected_beta)
    
    beta_diff = theta_beta_rot - expected_beta
    beta_diff_mean = np.mean(beta_diff, axis=1)
    
    print(np.mean(beta_diff_mean))
    
   # plot the data
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(1,9):
        ax.scatter(Y_beta_rot[:,i-1], Z_beta_rot[:,i-1],
                   s=10)
        ax.set_aspect('equal')
    plt.show()
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,10))
    for i in range(1,9):
        ax.scatter(np.deg2rad(theta_beta_rot[:,i-1]), r_beta_rot[:,i-1],
                   s=10)
    plt.show()
    
    
        
    
    
            
    # fig, ax = plt.subplots(figsize=(10,10))
    # for i in range(1,9):
    #     ax.scatter(Y_rot[:,i-1], Z_rot[:,i-1],
    #                s=10)
    #     ax.set_aspect('equal')
    # ax.scatter(0, 0, c='r')
    # ax.scatter(8 * np.cos(np.deg2rad(expected_theta)), 8 * np.sin(np.deg2rad(expected_theta)), c='r')
    # ax.set_xlabel('Y (mm)')
    # ax.set_ylabel('Z (mm)')
    
    # fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})
    # for i in range(1,9):
    #     ax.scatter(np.deg2rad(theta_rot[:,i-1]), r_rot[:,i-1],
    #                s=10)
    
    # fig, ax = plt.subplots(figsize=(10,10))
    # for i in range(1,9):
    #     ax.scatter(Y_df.loc[:, 'Y_' + str(i)], Z_df.loc[:, 'Z_' + str(i)],
    #                s=10)
    #     ax.set_aspect('equal')
    # ax.scatter(cntrs['y_centre'], cntrs['x_centre'], s=100, c='r')
    # ax.set_xlabel('Y (mm)')
    # ax.set_ylabel('Z (mm)')
    
    # fig, ax = plt.subplots(figsize=(10,10))
    # for i in range(1,9):
    #     ax.scatter(Y_corr[:,i-1], Z_corr[:,i-1],
    #                s=10)
    #     ax.set_aspect('equal')
    # ax.set_xlabel('Y (mm)')
    # ax.set_ylabel('Z (mm)')
    
    
    
    
    # print(expected_theta)
    # print(theta_rot[:10, :])    
    # theta_diff = theta_rot - expected_theta
    # theta_diff_mean = np.mean(theta_diff, axis=1)
    
    # print(np.mean(theta_diff_mean, axis=0))
    # print(8 * np.sin(np.deg2rad(np.mean(theta_diff_mean, axis=0))))
    
    # fig, ax = plt.subplots()
    # ax.scatter(Z_df.index, theta_diff_mean, c='b', s=10)
    # ax.axhline(np.mean(theta_diff_mean), c='r', linewidth=3)
    # ax.axhline(np.mean(theta_diff_mean)+np.std(theta_diff_mean), c='r', 
    #            linewidth=3, linestyle='--')
    # ax.axhline(np.mean(theta_diff_mean)-np.std(theta_diff_mean), c='r',
    #             linewidth=3, linestyle='--')
    # ax.set_xlabel('FPU ID')
    # ax.set_ylabel('Mean Theta Difference (deg)')
    # plt.legend()
    # plt.show()
    
    
    
    
    
    
    

    