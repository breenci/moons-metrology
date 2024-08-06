import matplotlib.pyplot as plt
import matplotlib as mpl
from src.metro_io import read_fibmap, read_metro_raw
import pandas as pd

# plt.style.use('seaborn-v0_8-colorblind')

mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14

# fibmap_fn = "data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_07/FPUCAL_07_01/FPU_Fiber_Mapping.txt"
# fltrd_fn ="data/FPU_calibrations/FPUCAL_MAY24/A7_B1_bad_movement/Coordinates_file_filtered.txt"

fibmap_fn = "data/PAE/xswitch_test/SPIE/FPU_Fiber_Mapping.txt"
fltrd_fn = "data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_01.04_METR_DAT/Coordinates_file_filtered.txt"

mode = 'clean'
fibmap = pd.read_csv(fibmap_fn, sep=' ', header=None, skiprows=18)
print(fibmap.head())
fltrd = read_metro_raw(fltrd_fn)

fig2, ax = plt.subplots(1, 1, figsize=(11, 11))
ax.plot(fibmap.iloc[:,2], fibmap.iloc[:, 3], '+', color='black', label='Centers')
ax.scatter(fibmap.iloc[:, 14], fibmap.iloc[:, 15], color='C0', label='Fibers',
        s=15, zorder=2)
for i in range(fibmap.shape[0]):
    x = [fibmap.iloc[i, 2], fibmap.iloc[i, 5], fibmap.iloc[i, 14]]
    y = [fibmap.iloc[i, 3], fibmap.iloc[i, 6], fibmap.iloc[i, 15]]
    ax.plot(x, y, color='C1', zorder=1, label='FPU Arms' if i == 0 else None)
if mode == 'full':
    # ax.plot(fltrd.iloc[:, 2], fltrd.iloc[:, 3], '.', color='black', zorder=0)
    ax.plot(fibmap.iloc[:, 8], fibmap.iloc[:, 9], '.', color='C3', markersize=5)
    ax.plot(fibmap.iloc[:, 11], fibmap.iloc[:, 12], '.', color='C4', markersize=5)
    ax.plot(fibmap.iloc[:, 5], fibmap.iloc[:, 6], '*', color='C2', markersize=5)

ax.set_xlabel('Y [mm]')
ax.set_ylabel('Z [mm]')
ax.grid()
ax.legend()
fig2.tight_layout()
fig2.savefig('documentation/SPIE/presentation/dectection.png', dpi=400)
plt.show()