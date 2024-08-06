import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from src.pnt_filter import points_in_box
from src.metro_io import read_metro_raw
from src.align import align_measurements, kabsch_umeyama
from src.get_corr import get_corr
from src.transform.spatial import matrix_transform

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.titlesize'] = 14

met_fn = "data/PAE/metro_accuracy/metro_accuracy_MOUT.txt"
coded_fn = "data/FULLPLATE_250923/FULLPLATE_01_01_01_TEM/FULLPLATE_01_01_01_TEM_HIN.txt"

met = np.loadtxt(met_fn)
coded = np.loadtxt(coded_fn)

nom_fid_fn = "data/PAE/metro_accuracy/fid_nominal_pos.csv"
nom_fid = pd.read_csv(nom_fid_fn)

built_fid_fn = "data/PAE/metro_accuracy/fid_as_built.csv"
built_fid = pd.read_csv(built_fid_fn).to_numpy()
built_fid = np.hstack((np.zeros((len(built_fid), 1)), built_fid))

bfid_c_idx = 0
# bfid_p_idx = 16
bfid_p_idx = 8

al_met_pos = align_measurements(coded, met, np.arange(1, 200))
al_met = np.copy(met)
al_met[:,1:4] = al_met_pos
al_met[:,1] = -1*al_met[:,1]

df = pd.DataFrame(columns=['FID ID', 'X', 'Y', 'Z', 'err_x', 'err_y', 'err_z', 'S', 'diff_x', 'diff_y', 'diff_z'])
for i in range(len(nom_fid)):
    fid_pos = nom_fid.iloc[i]
    # lengths for box filter
    L = [5, 25, 25]
    # # nominal centre of the mask
    box_centre = fid_pos[['X', 'Y', 'Z']].values
    
    # reference point for box filter
    fid_point = [box_centre[0] - L[0]/2, box_centre[1] - L[1]/2, 
                 box_centre[2] - L[2]/2]

    # do the filtering to get mask points
    fid_idx = points_in_box(al_met[:,1:4], fid_point, L)
    fid = al_met[fid_idx]
    fid = fid[(fid[:,7] > 1) & (fid[:,7] < 4)]
    
    np.savetxt(f'data/PAE/metro_accuracy/fid_{i}.txt', fid)
    
    # get the centre and point nearest to the centre
    fid_cntrd = fid[:, 1:4] - box_centre
    fid[:,1:4] = fid_cntrd
    fid_dist = np.linalg.norm(fid_cntrd, axis=1)
    fid_nearest = np.argsort(fid_dist)
    fid_centre = fid[fid_nearest[0]]
    fid_point = fid[fid_nearest[1]]

    
    a, b = get_corr(built_fid[:,0:3], fid[:, 1:4], (bfid_c_idx, bfid_p_idx),
                    (fid_nearest[0], fid_nearest[1]))
    
    built_fid_srtd = built_fid[a]
    fid_srtd = fid[b]
       
    trans_mat = kabsch_umeyama(built_fid_srtd[:,0:3], fid_srtd[:,1:4], scale=False)
    transformed_fid = matrix_transform(fid_srtd[:,1:4], trans_mat)
    
    diff = transformed_fid - built_fid_srtd[:,0:3]
    id = i * np.ones((len(diff), 1))
    
    fid_data = np.hstack((id, fid_srtd[:,1:8], diff))
    df = pd.concat([df, pd.DataFrame(fid_data, columns=df.columns)], ignore_index=True)

    
df['FID ID'] = df['FID ID'].astype(int)
df['R_diff'] = np.sqrt(df['diff_x']**2 + df['diff_y']**2 + df['diff_z']**2)
df['R_err'] = np.sqrt(df['err_x']**2 + df['err_y']**2 + df['err_z']**2)

df.to_csv('data/PAE/metro_accuracy/fiducial_errors.csv', index=False)

print(df['diff_x'].std())
print(df['diff_y'].std())
print(df['diff_z'].std())
# do some stats
# get the max diff y value


fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
flat_ax = ax.flatten()
fig.subplots_adjust(hspace=0, wspace=0)
for key, grp in df.groupby('FID ID'):
    scat = flat_ax[key].scatter(grp['Y'], grp['Z'], c=grp['diff_y'], s=500, vmin=df['diff_y'].min(), 
                           vmax=df['diff_y'].max())
    flat_ax[key].text(-7.5, 7, f'Measured Mean Error: {grp["diff_y"].abs().mean():.3f} mm\nPredicted Mean Error: {grp["err_y"].mean():.3f} mm')
    flat_ax[key].set_xlim(-8, 9)
    flat_ax[key].set_ylim(-8, 9)
flat_ax[2].set_xlabel('Y [mm]')
flat_ax[3].set_xlabel('Y [mm]')
flat_ax[0].set_ylabel('Z [mm]')
flat_ax[2].set_ylabel('Z [mm]')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.1, 0.04, 0.8])
fig.colorbar(scat, cax=cbar_ax, label='Error (mm)')
fig.suptitle('Measured Fiducial Errors - Y Axis')


fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
flat_ax = ax.flatten()
fig.subplots_adjust(hspace=0, wspace=0)
for key, grp in df.groupby('FID ID'):
    scat = flat_ax[key].scatter(grp['Y'], grp['Z'], c=grp['diff_z'], s=500, 
                                vmin=df['diff_z'].min(),
                            vmax=df['diff_z'].max())
    flat_ax[key].text(-7.5, 7, f'Measured Mean Error: {grp["diff_z"].abs().mean():.3f} mm\nPredicted Mean Error: {grp["err_z"].mean():.3f} mm')
    flat_ax[key].set_xlim(-8, 9)
    flat_ax[key].set_ylim(-8, 9)
flat_ax[2].set_xlabel('Y [mm]')
flat_ax[3].set_xlabel('Y [mm]')
flat_ax[0].set_ylabel('Z [mm]')
flat_ax[2].set_ylabel('Z [mm]')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.1, 0.04, 0.8])
fig.colorbar(scat, cax=cbar_ax, label='Error (mm)')
fig.suptitle('Measured Fiducial Errors - Z Axis', fontsize=16)

fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
flat_ax = ax.flatten()
fig.subplots_adjust(hspace=0, wspace=0)
for key, grp in df.groupby('FID ID'):
    scat = flat_ax[key].scatter(grp['Y'], grp['Z'], c=grp['R_diff'], s=500, 
                                vmin=df['R_diff'].min(),
                            vmax=df['R_diff'].max())
    flat_ax[key].text(-7.5, 7, f'Measured Mean Error: {grp["R_diff"].mean():.3f} mm\nPredicted Mean Error: {grp["R_err"].mean():.3f} mm')
    flat_ax[key].set_xlim(-8, 9)
    flat_ax[key].set_ylim(-8, 9)
flat_ax[2].set_xlabel('Y [mm]')
flat_ax[3].set_xlabel('Y [mm]')
flat_ax[0].set_ylabel('Z [mm]')
flat_ax[2].set_ylabel('Z [mm]')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.1, 0.04, 0.8])
fig.colorbar(scat, cax=cbar_ax, label='Error (mm)')
# fig.suptitle('Measured Fiducial Errors', fontsize=16)
fig.savefig('documentation/SPIE/paper/fiducial_errors.png', dpi=400)


plt.show()