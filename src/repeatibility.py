"""
This script is used to calculate the repeatibility of FPU measurements.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from src.metro_io import read_metro_out, read_FPU_out, read_fibmap
from src.pltpolar2pltcart import polar2cart
import pandas as pd
from src.get_corr import match_near_nbrs


# define a function to read and match raw hexagon output
def read_hexout(fn, exp_pos_fn, dot_type=2.6):
    # load the data
    raw_data = np.loadtxt(fn)
    
    # get the fibre dots
    fib_dots = raw_data[raw_data[:,7] == dot_type]
    # load the expected positions
    exp_pos = read_FPU_out(exp_pos_fn).sort_values(by='ID', ignore_index=True)
    # convert to cartesian
    exp_X, exp_Y, exp_Z = polar2cart(exp_pos['R'], np.deg2rad(exp_pos['Theta']), 4101.1)
    exp_cart = np.vstack((exp_X, exp_Y, exp_Z)).T
    exp_idx, raw_idx = match_near_nbrs(exp_cart, fib_dots[:,1:4], 2)
    
    fib_sorted = fib_dots[raw_idx, 1:4]
    data_out = pd.DataFrame(fib_sorted, columns=['X', 'Y', 'Z'])
    data_out['ID'] = exp_pos['ID'].iloc[exp_idx]
    
    return data_out

# define the file names
# ref_fn is the file to which the fn_list files are compared
ref_fn = 'data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_01.01_METR_DAT/FPU_polar_coordinates_01_01.txt'
fn_list = ['data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_01.01_METR_DAT/FPU_polar_coordinates_01_01.txt']
# exp_pos_fn is the file containing the expected positions of the FPU for 
# the HEXOUT data
exp_pos_fn = "data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_01.01_FPU_CONTROL.txt"

# define the type of data
ref_type = "METOUT"
measurement_type = "METOUT"

# load the reference data. Different methods for different data types
if ref_type == "METOUT":
    ref_data = read_metro_out(ref_fn).sort_values(by='ID', ignore_index=True)

    # convert to cartesian
    ref_X, ref_Y, ref_Z = polar2cart(ref_data['R'], np.deg2rad(ref_data['Theta']), 4101.4)
    
elif ref_type == "FIBMAP":
    ref_data = read_fibmap(ref_fn).sort_values(by='ID', ignore_index=True)
    ref_X = ref_data['X']
    ref_Y = ref_data['Y']
    ref_Z = ref_data['Z']
    
elif ref_type == "HEXOUT":
    # check if a exp_pos_fn is provided
    if exp_pos_fn is not None:
        # load the data
        ref_data = read_hexout(ref_fn, exp_pos_fn)
        
        ref_X = ref_data['X']
        ref_Y = ref_data['Y']
        ref_Z = ref_data['Z']
    
    else:
        raise ValueError("Expected positions file required for HEXOUT data.")
        
# stack the reference data
ref_cart = np.vstack((ref_X, ref_Y, ref_Z)).T

# initialise a data cube to store the data
diff_cube = np.zeros((ref_cart.shape[0], ref_cart.shape[1], len(fn_list)))

# open the files
for fn in fn_list:
    # metro_data = read_metro_out(fn).sort_values(by='ID', ignore_index=True)
    if measurement_type == "METOUT":
        measurement_data = read_metro_out(fn).sort_values(by='ID', ignore_index=True)
        
        # convert to cartesian
        m_X, m_Y, m_Z = polar2cart(measurement_data['R'], np.deg2rad(measurement_data['Theta']), 4101.4)
    
        
    elif measurement_type == "FIBMAP":
        measurement_data = read_fibmap(fn).sort_values(by='ID', ignore_index=True)
        m_X = measurement_data['X']
        m_Y = measurement_data['Y']
        m_Z = measurement_data['Z']
        
    elif measurement_type == "HEXOUT":
    # check if a exp_pos_fn is provided
        if exp_pos_fn is not None:
            measurement_data = read_hexout(fn, exp_pos_fn)
            
            m_X = measurement_data['X']
            m_Y = measurement_data['Y']
            m_Z = measurement_data['Z']
        else:
            raise ValueError("Expected positions file required for HEXOUT data.")
    
    # stack the measurement data
    m_data = np.vstack((m_X, m_Y, m_Z)).T
    
    # calculate the difference
    diff = m_data - ref_cart
    
    # store the difference
    diff_cube[:,:,fn_list.index(fn)] = diff
    
    
# --------------- Plotting ----------------
# plot the difference in the X, Y and Z directions
fig_fpu, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
for i in range(diff_cube.shape[2]):
    axes[0].scatter(ref_data['ID'], diff_cube[:,0,i], label='X')
    axes[1].scatter(ref_data['ID'], diff_cube[:,1,i], label='Y')
    axes[2].scatter(ref_data['ID'], diff_cube[:,2,i], label='Z')
for ax in axes:
    ax.set_ylabel('Difference (mm)')
    ax.set_xlim(ref_data['ID'].iloc[0], ref_data['ID'].iloc[-1])
    ax.grid()
axes[2].set_xlabel('FPU ID')

# plot the differrence in space (Y, Z plane)
fig_space, ax = plt.subplots(figsize=(10, 10))
for i in range(diff_cube.shape[2]):
    ax.scatter(diff_cube[:,1,i], diff_cube[:,2,i], label=fn_list[i])
# plot the mean and max difference
mean_err = np.mean(np.abs(diff_cube), axis=2)
max_err = np.max(np.abs(diff_cube), axis=2)

fig_mean, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for i in range(3):
    scat_mean = ax[i].scatter(m_Y, m_Z, c=mean_err[:,i], cmap='viridis', 
                  vmin=np.min(mean_err), vmax=np.max(mean_err))
    ax[i].set_xlabel('Y (mm)')
    ax[i].set_title(['Mean X', 'Mean Y', 'Mean Z'][i])
ax[0].set_ylabel('Z (mm)')
fig_mean.subplots_adjust(right=0.85)
cbar_ax = fig_mean.add_axes([0.9, 0.1, 0.04, 0.8])
fig_mean.colorbar(scat_mean, cax=cbar_ax, label='Error ($\mu m$)')


fig_max, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    scat_max = ax[i].scatter(m_Y, m_Z, c=max_err[:,i], cmap='viridis',
                  vmin=np.min(max_err), vmax=np.max(max_err))
    ax[i].set_xlabel('Y (mm)')
    ax[i].set_title(['Max X', 'Max Y', 'Max Z'][i])
ax[0].set_ylabel('Z (mm)')
fig_max.subplots_adjust(right=0.85)
cbar_ax = fig_max.add_axes([0.9, 0.1, 0.04, 0.8])
fig_max.colorbar(scat_max, cax=cbar_ax, label='Error ($\mu m$)')
plt.show()