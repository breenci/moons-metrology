"""Script for metrology calibrations"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.metro_io import read_metro_out, read_FPU_out
from src.transform.plate import polar2cart

if __name__ == '__main__':

    
    mpath_list = []
    spath_list = []
    alpha_list = [-180, -180, -180, 0] 
    beta_list = [6.5, 16.5, -3.5, 6.5]
    for i in [1, 2, 4, 5]:
        mpath_list.append('data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_01.0'+str(i)+'_METR_DAT/FPU_polar_coordinates_01_0'+str(i)+'.txt')
        spath_list.append('data/FPU_calibrations/FPUCAL_MAR24/FPUCAL_01.0'+str(i)+'_FPU_CONTROL.txt')
        
    fig, ax = plt.subplots()
    for n, i in enumerate(mpath_list):
        met_data = read_metro_out(i)
        sw_data = read_FPU_out(spath_list[n])
        
        cart_met = polar2cart(met_data['R'], np.deg2rad(met_data['Theta']), 4101.1)
        cart_met_arr = np.array(cart_met).T
        
        cart_sw = polar2cart(sw_data['R'], np.deg2rad(sw_data['Theta']), 4101.1)
        cart_sw_arr = np.array(cart_sw).T
        
        diff = cart_met_arr - cart_sw_arr
        
        blabel = r"$\beta$ = "+str(beta_list[n])+r'$^{\circ}$'
        alabel = r"$\alpha$ = "+str(alpha_list[n])+r'$^{\circ}$'
        
        ax.scatter(diff[:,1], diff[:,2], label=alabel+', '+blabel, s=5)
    ax.grid()
    ax.set_xlabel('Y Difference (mm)')
    ax.set_ylabel('Z Difference (mm)')
    plt.legend()    
    plt.show()
    
    

    

    
    
    
    