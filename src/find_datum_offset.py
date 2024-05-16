import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.metro_io import read_metro_out, read_FPU_out


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


def read_intermediate_data(fn):    
    # load the data
    pos_df = pd.read_csv(fn).sort_values(by='fpu_id', ignore_index=True)
    pos_df = pos_df.rename(columns={'fpu_id': 'ID', 'beta_r': 'R', 'beta_theta': 'Theta'})
    
    return pos_df


def build_pos_df(fn_list, mode='alpha'):
    '''
    INPUTS:
    - fn_list : List of file names (list)
    
    OUTPUTS:
    - pos_df  : Dataframe of the positions (pd.DataFrame)
    '''
    # intialise arrays to store the data
    Z_dict = {}
    Y_dict = {}
    
    # loop through the files and store the data in a dictionary
    for fn in fn_list:
        if mode == 'alpha':
        # get the alpha position from the file name
            folder_name = fn.split('/')[-2]
            pos = folder_name.split('_')[2]
            
            pos_df = read_intermediate_data(fn)
        
        if mode == 'beta':
            folder_name = fn.split('/')[-1]
            pos = folder_name.split('_')[4][:2]
        
            full_beta_df = read_FPU_out(fn)
            pos_df = full_beta_df[['ID', 'R', 'Theta']]
        
        Z_dict['fpu_ids'] = pos_df['ID']
        Y_dict['fpu_ids'] = pos_df['ID']
        
        # # make sure that the fpu ids are the same in each file
        if not np.array_equal(pos_df['ID'], Z_dict['fpu_ids']):
            raise ValueError("FPU IDs are not the same in each file")

        # store the data in the arrays
        Z_dict["Z_" + pos], Y_dict["Y_" + pos] = convert_to_cartesian(pos_df['R'], pos_df['Theta'])
        
    # convert the dictionary to a dataframe
    Y_df = pd.DataFrame(Y_dict).set_index('fpu_ids')
    Z_df = pd.DataFrame(Z_dict).set_index('fpu_ids')
    
    return Y_df, Z_df


def find_offset(fn_list, centres_fn, expected_angles, alpha_0=142, mode='alpha',
                arm_length_fn=None):
    '''
    INPUTS:
    - fn_list         : List of file names (list)
    - centres_fn      : File name of the centres (str)
    - expected_angles : Expected angles (np.array)
    - alpha_0         : Alpha 0 angle (float)
    
    OUTPUTS:
    - beta_diff_mean  : Mean difference in beta angles (np.array)
    '''
    # load the data
    cntrs = pd.read_csv(centres_fn).sort_values(by='FPU', ignore_index=True)
    # TODO: remove this for the actual centres
    cntrs['x_centre'] = -1*cntrs['x_centre']
    
    # extract the xy posistions of the rotation points from the data
    Y_df, Z_df = build_pos_df(fn_list, mode=mode)
    
    # subtract the centre positions
    Y_corr = Y_df.to_numpy() - cntrs['y_centre'].to_numpy().reshape(-1,1)
    Z_corr = Z_df.to_numpy() - cntrs['x_centre'].to_numpy().reshape(-1,1)
    
    # rotate each set of YZ data by alpha offset and cobvert to polar
    Y_rot = np.zeros(Y_corr.shape)
    Z_rot = np.zeros(Z_corr.shape)
    r_rot = np.zeros(Y_corr.shape)
    theta_rot = np.zeros(Z_corr.shape)
    
    # for each position rotate the data
    for i in range(Y_corr.shape[1]):
        if type(alpha_0) == int:
            Y_rot[:,i], Z_rot[:,i] = rotate_data(Y_corr[:,i], Z_corr[:,i], alpha_0)
        if type(alpha_0) == np.ndarray:
            for j in range(Y_corr.shape[0]):
                Y_rot_arr, Z_rot_arr = rotate_data(Y_corr[j, i], Z_corr[j, i], alpha_0[j])
                Y_rot[j, i] = Y_rot_arr[0]
                Z_rot[j, i] = Z_rot_arr[0]
                
    if mode == 'beta':
        arm_lengths = pd.read_csv(arm_length_fn).sort_values(by='fpu_id', ignore_index=True)
        L_alpha = arm_lengths['alpha_length']
        
        # Y_beta_rot = Y_beta_rot + L_alpha.to_numpy().reshape(-1,1)
        Y_rot = Y_rot + 8
    
    for i in range(Y_corr.shape[1]):
        r_rot[:,i], theta_rot[:,i] = convert_to_polar(Y_rot[:,i], Z_rot[:,i], wrap=True)
    
    theta_diff = theta_rot - expected_angles
    
    # plot the data
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(1,9):
        ax.scatter(Y_rot[:,i-1], Z_rot[:,i-1],
                   s=10)
        ax.set_aspect('equal')
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,10))
    for i in range(1,9):
        ax.scatter(np.deg2rad(theta_rot[:,i-1]), r_rot[:,i-1],
                   s=10)
    plt.show()
    
    return theta_diff
    
    
if __name__ == "__main__":
    # get the data
    inter_data_fns = sorted(glob.glob("data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/**/arm_length_intermediate_data.csv", recursive=True))
    beta_data_fns = sorted(glob.glob("data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/ALPHA_POSITION_1_2024-05-05T160410/*txt"))
    
    print(beta_data_fns)
    cntrs_fn = "data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/fpu_centres.csv"
    arm_length_fn='data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/arm_length_data.csv'
    
    expected_theta = np.array([-180., -137.625, -95.25, -52.875, -10.5, 31.875, 
                               74.25, 116.625])
    expected_beta = np.array([-172., -132.125, -92.25, -52.375, -12.5, 27.375, 
                              67.25, 107.125])
    
    # wrap the angles
    expected_theta[expected_theta < 0] += 360
    expected_beta[expected_beta < 0] += 360
    
    # TODO: alpha0 should be an array
    alpha0 = 142
    theta_diff = find_offset(inter_data_fns, cntrs_fn, expected_theta)
                             
    theta_diff_mean = np.mean(theta_diff, axis=1)
    alpha_offset = theta_diff_mean + alpha0
    

    beta_diff = find_offset(beta_data_fns, cntrs_fn, expected_beta, mode='beta',
                            arm_length_fn=arm_length_fn, alpha_0=alpha_offset)
    
    beta_diff_mean = np.mean(beta_diff, axis=1)
        