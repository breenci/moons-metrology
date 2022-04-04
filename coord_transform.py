#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:06:23 2021

@author: ciaran
"""

 # %% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

plt.close('all')


def plane_fitter(point_coords):
    '''Fit a plane to a set of points and return the unit normal.'''

    # Subtract the centroid from the set of points
    centroid = np.mean(point_coords, axis=0)
    cntrd_pnts = point_coords - centroid
    
    # Perform singular value decomposition
    svd, _, _ = np.linalg.svd(cntrd_pnts.T)
    
    # Final column of the svd gives the unit normal
    norm = svd[:,2]
    
    # Take the +ve norm
    if norm[1]>0:
        norm = -1*norm
    
    return norm, centroid


def change_basis(point_coords, y_new, x_new):
    '''Performs a change of basis operation'''
    
    # normalise new unit vectors
    unit_y = y_new/np.sqrt(np.sum(y_new**2))
    unit_x = x_new/np.sqrt(np.sum(x_new**2))
    
    # find x unit vector normal to y and z
    unit_z = np.cross(unit_x, unit_y)
    
    # change of basis matrix
    S = np.vstack((unit_x,unit_y,unit_z))
    
    # change of basis calculation
    new_coords = np.matmul(S, point_coords.T).T
    
    return S, new_coords


def project_point_to_plane(normal, point, pln_pnt=(0,0,0)):
    ''' Projects a point to a plane'''
    
    #normalise vector
    normal = normal/np.sqrt(np.sum(normal**2))

    pln_to_pnt = point - pln_pnt
    
    # project to plane
    new_point = point - normal*np.dot(normal, pln_to_pnt)
    
    return new_point
    

# input origin and plane as pnts insetad of indices?
def transform_data(data, origin_ind, plane_ind, y0_ind):
    '''Performs the coordinate transformation.'''
    
    # Find unit normal by plane fitting
    plane_coords = data[plane_ind,:]
    unit_z,_ = plane_fitter(plane_coords)
    
    # Translate data to central AC mask
    new_origin = origin_ind #data[origin_ind,:]
    translated = data - new_origin
    
    # Define zero point for y axis rotaion
    y0 = translated[y0_ind,:]

    # project y0 onto plane centred at origin with normal parallel to z to
    # find y vector
    Vy = project_point_to_plane(unit_z,y0)
    
    S, transformed_coords = change_basis(translated, Vy, unit_z)
        
    return S, transformed_coords


def trans_mat_from_plane(data, origin_ind, plane_ind, y0_ind):

     # Find unit normal by plane fitting
    plane_coords = data[plane_ind,:]
    unit_x, pln_pnt = plane_fitter(plane_coords)


def rotate_coords(x, y, t_deg):
    '''Rotates coords in xy plane'''

    t = t_deg*np.pi/180.
    x_r = x*np.cos(t) - y*np.sin(t)
    y_r = y*np.cos(t) + x*np.sin(t)
    return x_r, y_r


if __name__ == '__main__':
    
    # # Read data from hexagon output. Input data is in fixed width file format.
    # hex_coords = pd.read_fwf('coord_files/nominal_coords_all_15092021.obc', usecols=range(1,8))
    
    # # extract xyz values as numpy array
    # coords_arr = hex_coords.iloc[:, 1:4].to_numpy()
    
    # # index of new origin point
    # origin_ind = 25
    
    # # extract index of points to be used for plane fitting. Here using targets
    # # on edge of plate. Extracted using y pos. Use indices in future?
    # Y = coords_arr[:,1]
    # X = coords_arr[:,0]
    # Z = coords_arr[:,2]
    # plane_ind = np.where((Y>-50) & (Y<-5))[0]
    
    # # index of point used to define y axis direction
    # y0_ind = 248
    
    # # do transformation and assign to pandas dataframe
    # S, new_coords = transform_data(coords_arr,origin_ind,plane_ind,y0_ind)
    
    # out_coords = hex_coords.assign(X=new_coords[:,0], Y=new_coords[:,1], 
    #                                Z=new_coords[:,2])


    # # save in input format (.obc)
    # header = '%-4s %-9s %-9s %-9s %-6s %-6s %-6s' % ('ID','X','Y','Z','errX',
    #                                                   'errY', 'errZ')
    # fmt = '  %-4i %-9.4f %-9.4f %-9.4f %-6.4f %-6.4f %-6.4f'

    measurement = 'datum2'

    filename = os.path.expanduser('~/Documents/moons_metrology/caltest_22_06_21/measurements/uncalibrated/'+measurement+'.txt')
    pnt_cld = np.loadtxt(filename)
    coords_arr = pnt_cld[:,1:4]
    id = pnt_cld[:,0]

    plane_ind = np.where((id < 65))[0]
    y0_ind = np.where(id==62)[0][0]
    origin = np.array([0,0,0])

    S, new_coords = transform_data(coords_arr, origin, plane_ind, y0_ind)

    y,z = rotate_coords(new_coords[:,1], new_coords[:,2], -244.1015515)

    new_coords[:,1] = y
    new_coords[:,2] = z

    out_coords = np.copy(pnt_cld)
    out_coords[:,1:4] = new_coords

    out_file = os.path.expanduser('~/Documents/moons_metrology/caltest_22_06_21/transformed/uncalibrated/'+measurement+'_t.txt')
    #np.savetxt(out_file, out_coords)


    #np.savetxt('transformed_coords_new_def.obc', out_coords, fmt = fmt, header=header)
    
    # plotting
    fig1 = plt.figure(figsize=(12,12))
    ax1 = fig1.add_subplot(121)
    # ax1.scatter(coords_arr[:28,0], coords_arr[:28,1], s=10)
    # ax1.scatter(coords_arr[28:78,0], coords_arr[28:78,1], s=1)
    # ax1.scatter(coords_arr[78:,0], coords_arr[78:,1], s=10)
    # ax1.scatter(coords_arr[12,0], coords_arr[12,1], s=15)
    ax1.scatter(coords_arr[:,1], coords_arr[:,0], s=1)
    ax1.axhline(0,linestyle = '--', color = 'black')
    ax1.axvline(0, linestyle = '--', color = 'black')
    ax1.set_xlabel('Y (mm)')
    ax1.set_ylabel('X (mm)')
    ax1.set_title('Before Transformation')
    ax2 = fig1.add_subplot(122)
    # ax2.scatter(new_coords[:28,0], new_coords[:28,1], s=10)
    # ax2.scatter(new_coords[28:78,0], new_coords[28:78,1], s=1)
    # ax2.scatter(new_coords[78:,0], new_coords[78:,1], s=10)
    # ax2.scatter(new_coords[12,0], new_coords[12,1], s=15)
    ax2.scatter(new_coords[:,1], new_coords[:,0], s=10)
    ax2.axhline(0,linestyle = '--', color = 'black')
    ax2.axvline(0, linestyle = '--', color = 'black')
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('X (mm)')
    ax2.set_title('After Transformation')
    plt.show()


    fig2 = plt.figure(figsize=(12,12))
    ax1 = fig2.add_subplot(121)
    # ax1.scatter(coords_arr[:28,0], coords_arr[:28,1], s=10)
    # ax1.scatter(coords_arr[28:78,0], coords_arr[28:78,1], s=1)
    # ax1.scatter(coords_arr[78:,0], coords_arr[78:,1], s=10)
    # ax1.scatter(coords_arr[12,0], coords_arr[12,1], s=15)
    ax1.scatter(coords_arr[:,1], coords_arr[:,2], s=1)
    ax1.axhline(0,linestyle = '--', color = 'black')
    ax1.axvline(0, linestyle = '--', color = 'black')
    ax1.set_xlabel('Y (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title('Before Transformation')
    ax2 = fig2.add_subplot(122)
    # ax2.scatter(new_coords[:28,0], new_coords[:28,1], s=10)
    # ax2.scatter(new_coords[28:78,0], new_coords[28:78,1], s=1)
    # ax2.scatter(new_coords[78:,0], new_coords[78:,1], s=10)
    # ax2.scatter(new_coords[12,0], new_coords[12,1], s=15)
    ax2.scatter(new_coords[:,1], new_coords[:,2], s=1)
    ax2.axhline(0,linestyle = '--', color = 'black')
    ax2.axvline(0, linestyle = '--', color = 'black')
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('After Transformation')
    plt.show()
    
    
# %%

