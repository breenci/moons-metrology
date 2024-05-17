import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.pnt_filter import points_in_box

# load the nominal mask posistions
pnt_cloud_fn = 'data/ACQCAL/ACQCAL_01_01_METR_RAW.txt'
pnt_cld = np.loadtxt(pnt_cloud_fn)

# flip the x axis to make the coordinate system left handed
pnt_cld[:,1] = -1*pnt_cld[:,1]

exp_mask_pos = pd.read_csv('data/ACQCAL/AC_nominal_positions.csv')
# remove masks 18, 11, 5 (these are not fitted)
rm_list = [18, 11, 5]
mask_pos = exp_mask_pos[~exp_mask_pos['AC ID'].isin(rm_list)]

# get the masks and store each set of points in a dictionary
mask_points = {}

# For each mask find the height difference
fig, ax = plt.subplots()
for i in mask_pos['AC ID'].values:
    # select the mask
    ac = mask_pos[mask_pos['AC ID'] == i]
    
    # get the points in the mask
    # lengths for box filter
    L = [10, 27, 27]
    # nominal centre of the mask
    box_centre = ac[['X', 'Y', 'Z']].values[0]
    # reference point for box filter
    box_point = [box_centre[0] - L[0]/2, box_centre[1] - L[1]/2, 
                    box_centre[2] - L[2]/2]

    # do the filtering to get mask points
    ac_mask = points_in_box(pnt_cld[:,1:4], box_point, L)
    ac_pnts = pnt_cld[ac_mask]
    
    # remove any bad detections (too small)
    ac_pnts = ac_pnts[ac_pnts[:,7] > 1]
    
    ax.scatter(ac_pnts[:,2], ac_pnts[:,3], s=1)
    
    mask_points[i] = ac_pnts
    
plt.show()
    
    