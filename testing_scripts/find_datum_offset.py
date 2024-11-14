import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.metro_io import read_metro_out
import argparse
import os


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

def get_beta_pos(fn):
    
    folder_name = fn.split('/')[-1]
    pos = folder_name.split('_')[4].split('.')[0][1:]
    print(pos)
    return int(pos)
    


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
            pos = folder_name.split('_')[4].split('.')[0][1:]
            
        
            full_beta_df = read_metro_out(fn)
            pos_df = full_beta_df[['ID', 'R', 'Theta']]
            

        # store the data in the arrays
        z, y = convert_to_cartesian(pos_df['R'], pos_df['Theta'])
        
        # create a pandas series for x and y
        y_series = pd.Series(y.to_numpy(), name=pos, index=pos_df['ID'])
        z_series = pd.Series(z.to_numpy(), name=pos, index=pos_df['ID'])
        
        # store the series in the dictionary
        Y_dict[pos] = y_series
        Z_dict[pos] = z_series
        
    # convert the dictionary to a dataframe
    Y_df = pd.DataFrame(Y_dict)
    Z_df = pd.DataFrame(Z_dict)
    
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
    cntrs = pd.read_csv(centres_fn).sort_values(by='fpu_id', ignore_index=True)
    
    # convert to cartesian
    cntrs['x_centre'], cntrs['y_centre'] = convert_to_cartesian(cntrs['alpha_r'], cntrs['alpha_theta'])

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
        elif type(alpha_0) == np.ndarray:
            for j in range(Y_corr.shape[0]):
                Y_rot_arr, Z_rot_arr = rotate_data(Y_corr[j, i], Z_corr[j, i], alpha_0[j])
                Y_rot[j, i] = Y_rot_arr[0]
                Z_rot[j, i] = Z_rot_arr[0]                   
        else:
            raise ValueError("alpha_0 must be an integer or an array")
        
    if mode == 'beta':
        arm_lengths = pd.read_csv(arm_length_fn).sort_values(by='fpu_id', ignore_index=True)
        L_alpha = arm_lengths['alpha_length']
        
        Y_rot = Y_rot + L_alpha.to_numpy().reshape(-1,1)

    
    for i in range(Y_corr.shape[1]):
        r_rot[:,i], theta_rot[:,i] = convert_to_polar(Y_rot[:,i], Z_rot[:,i], wrap=False)
    
    theta_diff = theta_rot - expected_angles
    
    # convert to dataframe with fpu ids as index
    theta_diff_df = pd.DataFrame(theta_diff, columns=[str(exp) for exp in expected_angles], 
                                 index=Y_df.index)
    
    # TODO: Remove plots? This should be its own commit
    # plot the data
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(Y_corr.shape[1]):
        if mode == 'alpha':
            ax.title.set_text('Alpha')
        elif mode == 'beta':
            ax.title.set_text('Beta')
        ax.scatter(Y_rot[:,i], Z_rot[:,i],
                   s=1, label=str(expected_angles[i]) + '$^{\circ}$', alpha=0.1)
        if mode == 'alpha':
            ax.plot([0, 8*np.cos(np.deg2rad(expected_angles[i]))], 
                    [0, 8*np.sin(np.deg2rad(expected_angles[i]))],
                    linestyle='--')
        if mode == 'beta':
            ax.plot([0, 17*np.cos(np.deg2rad(expected_angles[i]))], 
                    [0, 17*np.sin(np.deg2rad(expected_angles[i]))],
                    linestyle='--')
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    ax.legend(markerscale=10, title='Expected Angle')
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,8))
    for i in range(Y_corr.shape[1]):
        ax.scatter(np.deg2rad(theta_rot[:,i]), r_rot[:,i],
                   s=1, label=str(expected_angles[i]) + '$^{\circ}$')
    if mode == 'alpha':
        ax.set_title('Alpha Offset', fontdict={'fontsize': 20})
        ax.set_rmax(10)
        ax.set_rticks([2, 4, 6, 8, 10])
    elif mode == 'beta':
        ax.set_title('Beta Offset', fontdict={'fontsize': 20})
        ax.set_rmax(20)
        ax.set_rticks([5, 10, 15, 20])
    ax.tick_params(labelsize=12)
    ax.legend(markerscale=10, title='Expected Angle', bbox_to_anchor=(1.2, 1), 
              title_fontsize=12)
    plt.tight_layout()
    return theta_diff_df
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex_alpha', type=float, nargs='+', default=None)
    parser.add_argument('--ex_beta', type=float, nargs='+', default=None)
    parser.add_argument('--alpha0', type=float, default=142)
    parser.add_argument('--beta0', type=float, default=6.5)
    args = parser.parse_args()
    
    # get the data intermediate for alpha, metrology output for beta
    inter_data_fns = sorted(glob.glob("data/PAE_closure/arm_cal/FPU_ARM_LENGTH_DATA_2024-11-11T190751/**/arm_length_intermediate_data.csv", recursive=True))
    beta_data_fns = sorted(glob.glob("data/PAE_closure/arm_cal/FPU_ARM_LENGTH_DATA_2024-11-11T190751/ALPHA_POSITION_0_2024-11-11T200858/*txt"), key=get_beta_pos)
    cntrs_fn = "data/PAE_closure/arm_cal/FPU_ARM_LENGTH_DATA_2024-11-11T190751/arm_length_data.csv"
    arm_length_fn='data/PAE_closure/arm_cal/FPU_ARM_LENGTH_DATA_2024-11-11T190751/arm_length_data.csv'
    
    alpha4beta_folder = beta_data_fns[0].split('/')[-2]
    alpha4beta_pos = int(alpha4beta_folder.split('_')[2])
    
    if args.ex_alpha is None:
        expected_alpha = np.array([-178.0, -131.143, -84.286, -37.429, 9.428, 56.285, 103.142, 149.999])
    else:
        expected_alpha = np.array(args.ex_alpha)
        
        
    if args.ex_beta is None:
        expected_beta = np.array([-165., -124.714, -84.428, -44.142, 11.144, 51.43, 91.716, 132.002])
    else:
        expected_beta = np.array(args.ex_beta)
    
    # wrap the angles
    unwrapped_alpha = expected_alpha.copy()
    unwrapped_beta = expected_beta.copy()
    # expected_alpha[expected_alpha < 0] += 360
    # expected_beta[expected_beta < 0] += 360
    
    
    alpha0 = args.alpha0
    alpha_diff = find_offset(inter_data_fns, cntrs_fn, expected_alpha)
    # alpha_diff.to_csv('data/PAE_closure/arm_cal/FPU_ARM_LENGTH_DATA_2024-11-11T190751/alpha_diff.csv')

    corr_alpha = find_offset(inter_data_fns, cntrs_fn, expected_alpha, 
                             alpha_0=alpha0-alpha_diff.iloc[:,0].to_numpy(), 
                             mode='alpha')
    
    beta_alpha0 = alpha0 - alpha_diff.iloc[:,alpha4beta_pos].to_numpy() + (-180 - expected_alpha[0])
    # remove the magic number
    beta_diff = find_offset(beta_data_fns, cntrs_fn, expected_beta, mode='beta',
                            arm_length_fn=arm_length_fn, 
                            alpha_0=beta_alpha0)
    # beta_diff.to_csv('data/PAE_closure/arm_cal/FPU_ARM_LENGTH_DATA_2024-11-11T190751/beta_diff.csv')
            
    beta_diff_mean = np.nanmean(beta_diff.to_numpy(), axis=1)
    # create the config file
    # load the centres file
    centres = pd.read_csv(cntrs_fn)
    config = pd.DataFrame({'fpu_id': centres['fpu_id'], 'alpha_r':centres['alpha_r'], 
                           'alpha_theta': centres['alpha_theta'],
                           'alpha_length': centres['alpha_length'],
                           'beta_length': centres['beta_length'],
                           'alpha_zero': -1*(alpha0-alpha_diff.iloc[:,0].to_numpy()),
                           'beta_zero': args.beta0-beta_diff_mean}) 
    # config.to_csv('data/PAE_closure/arm_cal/FPU_ARM_LENGTH_DATA_2024-11-11T190751/full_config.csv', index=False)
    
    
    # id of fpu to plot
    report_id = 447
    # plot the given fpu
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax[0].scatter(unwrapped_alpha, alpha_diff.loc[report_id], label='Alpha')
    ax[1].scatter(unwrapped_beta, beta_diff.loc[report_id], label='Beta')
    ax[0].set_title('FPU ID: ' + str(report_id))
    ax[1].set_xlabel('Expected Angle (degrees)')
    ax[0].set_ylabel('Alpha Difference (degrees)')
    ax[1].set_ylabel('Beta Difference (degrees)')
    ax[0].grid()
    ax[1].grid()
    # ax[0].legend()
    plt.show()