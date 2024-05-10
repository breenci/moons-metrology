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


def convert_to_polar(x, y):
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
    
    theta[theta < 0] += 2*np.pi  # convert to 0-360
    return (r, np.rad2deg(theta))


if __name__ == "__main__":
    # get the data
    inter_data_fns = sorted(glob.glob("data/FPU_calibrations/FPUCAL_MAY24/simulated_data/FPU_ARM_LENGTH_DATA_WITH_NOISE/**/arm_length_intermediate_data.csv", recursive=True))
    
    # load the data to a dataframe
    # intialise a list to store the data
    Cbeta_dict = {}
    
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
        else:
            if not np.array_equal(Cbeta_dict['fpu_ids'], Cbeta_df['fpu_id']):
                raise ValueError("The fpus ids are not the same in all the files")
        
        # store the data in the dictionary
        Cbeta_dict['r_' + alpha_pos] = Cbeta_df['beta_r']
        Cbeta_dict['theta_' + alpha_pos] = Cbeta_df['beta_theta']
        Cbeta_dict['Y_' + alpha_pos] = Y
        Cbeta_dict['Z_' + alpha_pos] = Z
        
    # convert the dictionary to a dataframe
    Cbeta_df = pd.DataFrame(Cbeta_dict)
    
    # plot fpu 1
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(1,8):
        ax.scatter(Cbeta_df.loc[:, 'Y_' + str(i)], Cbeta_df.loc[:, 'Z_' + str(i)],
                   s=10)
        ax.set_aspect('equal')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    plt.show()
    print(Cbeta_df)