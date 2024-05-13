import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob


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
    
    # load the data into numpy arrays
    # intialise arrays to store the data
    Cbeta_dict = {}
    Z_dict = {}
    Y_dict = {}

    
    # loop through the files and store the data in a dictionary
    for fn in inter_data_fns:
        # get the alpha position from the file name
        folder_name = fn.split('/')[-2]
        alpha_pos = folder_name.split('_')[2]
        
        # load the data
        full_df = pd.read_csv(fn)
        
        # extract the beta centre positions
        Cbeta_df = full_df[['fpu_id', 'beta_r', 'beta_theta']]
        Y, Z = convert_to_cartesian(Cbeta_df['beta_r'], Cbeta_df['beta_theta'])
        
        
        # sort the data by fpu_id
        Cbeta_df = Cbeta_df.sort_values(by='fpu_id', ignore_index=True)
        
        # check that the fpus are the same in all the files
        if 'fpu_ids' not in Cbeta_dict:
            Cbeta_dict['fpu_ids'] = Cbeta_df['fpu_id']
            Y_dict['fpu_ids'] = Cbeta_df['fpu_id']
            Z_dict['fpu_ids'] = Cbeta_df['fpu_id']
        else:
            if not np.array_equal(Cbeta_dict['fpu_ids'], Cbeta_df['fpu_id']):
                raise ValueError("The fpus ids are not the same in all the files")
        
        # store the data in the dictionary
        Cbeta_dict['r_' + alpha_pos] = Cbeta_df['beta_r']
        Cbeta_dict['theta_' + alpha_pos] = Cbeta_df['beta_theta']
        
        # store the data in the arrays
        Z_dict["Z_" + alpha_pos], Y_dict["Y_" + alpha_pos] = convert_to_cartesian(Cbeta_df['beta_r'], Cbeta_df['beta_theta'])
        
         
        
    # convert the dictionary to a dataframe
    Cbeta_df = pd.DataFrame(Cbeta_dict)
    Y_df = pd.DataFrame(Y_dict).set_index('fpu_ids')
    Z_df = pd.DataFrame(Z_dict).set_index('fpu_ids')
    
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(1,9):
        ax.scatter(Y_df.loc[:, 'Y_' + str(i)], Z_df.loc[:, 'Z_' + str(i)],
                   s=10)
        ax.set_aspect('equal')
    ax.scatter(cntrs['y_centre'], cntrs['x_centre'], s=100, c='r')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    
    # subtract the centre positions
    Y_corr = Y_df.to_numpy() - cntrs['y_centre'].to_numpy().reshape(-1,1)
    Z_corr = Z_df.to_numpy() - cntrs['x_centre'].to_numpy().reshape(-1,1)
    
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(1,9):
        ax.scatter(Y_corr[:,i-1], Z_corr[:,i-1],
                   s=10)
        ax.set_aspect('equal')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    
    # rotate each set of YZ data by 142 degrees
    Y_rot = np.zeros(Y_corr.shape)
    Z_rot = np.zeros(Z_corr.shape)
    for i in range(Y_corr.shape[1]):
        Y_rot[:,i], Z_rot[:,i] = rotate_data(Y_corr[:,i], Z_corr[:,i], 142)
        
    expected_theta = np.array([-180., -137.625, -95.25, -52.875, -10.5, 31.875, 74.25, 116.625])
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(1,9):
        ax.scatter(Y_rot[:,i-1], Z_rot[:,i-1],
                   s=10)
        ax.set_aspect('equal')
    ax.scatter(0, 0, c='r')
    ax.scatter(8 * np.cos(np.deg2rad(expected_theta)), 8 * np.sin(np.deg2rad(expected_theta)), c='r')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    
    # convert to polar coordinates
    r_rot = np.zeros(Y_rot.shape)
    theta_rot = np.zeros(Z_rot.shape)
    
    for i in range(Y_rot.shape[1]):
        r_rot[:,i], theta_rot[:,i] = convert_to_polar(Y_rot[:,i], Z_rot[:,i], wrap=False)
    
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})
    for i in range(1,9):
        ax.scatter(np.deg2rad(theta_rot[:,i-1]), r_rot[:,i-1],
                   s=10)
    
    
    
    # wrap the angles to 0-360
    theta_rot[theta_rot < 0] += 360
    expected_theta[expected_theta < 0] += 360
    
    print(expected_theta)
    print(theta_rot[:10, :])    
    theta_diff = theta_rot - expected_theta
    theta_diff_mean = np.mean(theta_diff, axis=1)
    
    print(np.mean(theta_diff_mean, axis=0))
    print(8 * np.sin(np.deg2rad(np.mean(theta_diff_mean, axis=0))))
    
    fig, ax = plt.subplots()
    # ax.scatter(Cbeta_df['fpu_ids'], theta_diff_mean, c='b', s=10)
    ax.scatter(Cbeta_df['fpu_ids'], theta_diff[:,0], c='r', s=10, 
               alpha=.2, label='Alpha=$-180^{\circ}$')
    ax.scatter(Cbeta_df['fpu_ids'], theta_diff[:,7], c='g', s=10, 
               alpha=.2, label='Alpha=$116.625^{\circ}$')
    ax.axhline(np.mean(theta_diff[:,0]), c='r', linewidth=3)
    ax.axhline(np.mean(theta_diff[:,7]), c='g', linewidth=3)
    ax.text(0, np.mean(theta_diff[:,0])-.03, f'Mean Theta Diff: {np.mean(theta_diff[:,0]):.3f}', color='r', fontsize=14)
    ax.text(0, np.mean(theta_diff[:,7])+.01, f'Mean Theta Diff: {np.mean(theta_diff[:,7]):.3f}', color='g', fontsize=14)
    ax.set_xlabel('FPU ID')
    ax.set_ylabel('Mean Theta Difference (deg)')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    

    