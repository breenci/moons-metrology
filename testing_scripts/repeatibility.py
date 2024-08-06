"""
This script is used to calculate the repeatibility of FPU measurements.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from src.metro_io import read_metro_out, read_FPU_out, read_fibmap
from src.transform.plate import polar2cart
import pandas as pd
from src.get_corr import match_near_nbrs
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14


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
    print(fib_dots[raw_idx, 7])
    data_out = pd.DataFrame(fib_sorted, columns=['X', 'Y', 'Z'])
    data_out['ID'] = exp_pos['ID'].iloc[exp_idx]
    
    return data_out

# define the file names
# ref_fn is the file to which the fn_list files are compared
ref_fn = 'data/FPU_calibrations/FPUCAL_MAY24/repeatability/run2/Coordinates_file_filtered.txt'
fn_list = ['data/FPU_calibrations/FPUCAL_MAY24/repeatability/run3/Coordinates_file_filtered.txt',
           'data/FPU_calibrations/FPUCAL_MAY24/repeatability/run4/Coordinates_file_filtered.txt',
           'data/FPU_calibrations/FPUCAL_MAY24/repeatability/run5/Coordinates_file_filtered.txt']
# ref_fn = 'data/FPU_calibrations/FPUCAL_MAY24/repeatability/run6/FPU_Fiber_Mapping.txt'
# fn_list = ['data/FPU_calibrations/FPUCAL_MAY24/repeatability/run7/FPU_Fiber_Mapping.txt',
#            'data/FPU_calibrations/FPUCAL_MAY24/repeatability/run8/FPU_Fiber_Mapping.txt',
#            'data/FPU_calibrations/FPUCAL_MAY24/repeatability/run9/FPU_Fiber_Mapping.txt']
# exp_pos_fn is the file containing the expected positions of the FPU for 
# the HEXOUT data
exp_pos_fn = "data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_01.01_FPU_CONTROL.txt"

# define the type of data
ref_type = "HEXOUT"
measurement_type = "HEXOUT"


# load the reference data. Different methods for different data types
if ref_type == "METOUT":
    ref_data = read_metro_out(ref_fn).sort_values(by='ID', ignore_index=True)

    # convert to cartesian
    ref_data['X'], ref_data['Y'], ref_data['Z'] = polar2cart(ref_data['R'], np.deg2rad(ref_data['Theta']), 4101.1)
    
    ref_r = np.sqrt(ref_data['X']**2 + ref_data['Y']**2 + ref_data['Z']**2)
    ref_r_df = pd.DataFrame({'ID': ref_data['ID'], 'R': ref_r}).set_index('ID')
    
elif ref_type == "FIBMAP":
    ref_data = read_fibmap(ref_fn).sort_values(by='ID', ignore_index=True)

    
elif ref_type == "HEXOUT":
    # check if a exp_pos_fn is provided
    if exp_pos_fn is not None:
        # load the data
        ref_data = read_hexout(ref_fn, exp_pos_fn)
        pos = ref_data.copy()[['ID', 'X', 'Y', 'Z']]

        ref_r = np.sqrt(ref_data['X']**2 + ref_data['Y']**2 + ref_data['Z']**2)
        ref_r_df = pd.DataFrame({'ID': ref_data['ID'], 'R': ref_r}).set_index('ID')
    
    else:
        raise ValueError("Expected positions file required for HEXOUT data.")
    

diff_list = []
# pos_list = [pos]
r_list = [ref_r_df]
for fn in fn_list:
    # metro_data = read_metro_out(fn).sort_values(by='ID', ignore_index=True)
    if measurement_type == "METOUT":
        measurement_data = read_metro_out(fn).sort_values(by='ID', ignore_index=True)
        
        # convert to cartesian
        measurement_data['X'], measurement_data['Y'], measurement_data['Z'] = polar2cart(measurement_data['R'], np.deg2rad(measurement_data['Theta']), 4101.4)
            
    elif measurement_type == "FIBMAP":
        measurement_data = read_fibmap(fn).sort_values(by='ID', ignore_index=True)
        
    elif measurement_type == "HEXOUT":
    # check if a exp_pos_fn is provided
        if exp_pos_fn is not None:
            measurement_data = read_hexout(fn, exp_pos_fn)
            
        else:
            raise ValueError("Expected positions file required for HEXOUT data.")
    
    
    m_pos = measurement_data.copy()[['ID', 'X', 'Y', 'Z']]
    # pos_list.append(m_pos)
    r = np.sqrt(measurement_data['X']**2 + measurement_data['Y']**2 + measurement_data['Z']**2)
    r_df = pd.DataFrame({'ID': measurement_data['ID'], 'R': r}).set_index('ID')
    r_list.append(r_df)
    # find the common IDs
    common_ids = np.intersect1d(ref_data['ID'], measurement_data['ID'])
    
    # filter the data
    ref_data_fltrd = ref_data[ref_data['ID'].isin(common_ids)].sort_values(by='ID', ignore_index=True)
    measurement_data_fltrd = measurement_data[measurement_data['ID'].isin(common_ids)].sort_values(by='ID', ignore_index=True)
    
    # check if the two data sets have the same IDs
    if not np.array_equal(ref_data_fltrd['ID'], measurement_data_fltrd['ID']):
        raise ValueError("FPU IDs are not the same in each file")
    
    # calculate the difference
    diff = measurement_data_fltrd[['X', 'Y', 'Z']] - ref_data_fltrd[['X', 'Y', 'Z']]
    diff['ID'] = ref_data_fltrd['ID']
    diff.set_index('ID', inplace=True, drop=False)
    diff_list.append(diff)


full_r = pd.concat(r_list, axis=1)
full_r.columns = ["R"+str(i) for i in range(len(r_list))] 

# do some stats
# std
std = full_r.std(axis=1)
    
# # --------------- Plotting ----------------
# # plot the difference in the X, Y and Z directions
fig_fpu, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
for i in range(len(diff_list)):
    axes[0].scatter(diff_list[i]['ID'], diff_list[i]['X'], label='X', s=1)
    axes[1].scatter(diff_list[i]['ID'], diff_list[i]['Y'], label='Y', s=1)
    axes[2].scatter(diff_list[i]['ID'], diff_list[i]['Z'], label='Z', s=1)
for ax in axes:
    ax.set_ylabel('Difference (mm)')
    ax.set_xlim(ref_data['ID'].iloc[0], ref_data['ID'].iloc[-1])
    ax.grid()
axes[2].set_xlabel('FPU ID')

# plot the differrence in space (Y, Z plane)
fig_space, ax = plt.subplots(figsize=(10, 10))
for i in range(len(diff_list)):
    ax.scatter(diff_list[i]['Y'], diff_list[i]['Z'], label="run "+str(i+1), s=1)
ax.set_xlabel('Y Difference (mm)')
ax.set_ylabel('Z Difference (mm)')
ax.grid()
plt.legend()

bins = np.arange(0, 0.01, .0005)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].hist(std, bins=bins)
ax[0].set_xlabel('RMS Standard Deviation [mm]')
ax[0].set_ylabel('No. of FPUs')
# plt.savefig('documentation/SPIE/presentation/repeat_hist.png')



# fig, ax = plt.subplots(figsize=(10, 10))
for i in range(full_r.shape[1]):
    ax[1].scatter(full_r.index, full_r.iloc[:,i]-full_r.mean(axis=1), s=10, 
               c='b', alpha=.3)
ax[1].set_xlim(full_r.index[0], full_r.index[-1])
ax[1].set_xlabel('FPU ID')
ax[1].set_ylabel('D$_{FPU}$ - mean(D$_{FPU}$) [mm]')
ax[1].grid()
plt.savefig('documentation/SPIE/presentation/repeat_all.png')
plt.tight_layout()
plt.show()
print(full_r.head())