"""Script to find the height difference between the AC mask and focal plane"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.preprocessing import sphere_filter
from src.pnt_filter import cylinder_filter
from src.transform.spatial import plane_fitter
from skspatial.objects import Line, Sphere


def sphereFit(spX,spY,spZ):
    """Fit a sphere to a set of points"""
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius[0], C[0][0], C[1][0], C[2][0]


def find_point_on_plane(A, B, C, D, known_coords, missing_coord='z'):
    """Find the value of the missing coordinate given the coefficients of the plane"""
    
    if missing_coord == 'z':
        x, y = known_coords
        z = (-A * x - B * y - D) / C
        missing_coord_val = z
        
    if missing_coord =='y':
        x, z = known_coords
        y = (-A * x - C * z - D) / B
        missing_coord_val = y
    
    if missing_coord =='x':
        y, z = known_coords
        x = (-B * y - C * z - D) / A
        missing_coord_val = x
        
    return missing_coord_val


def find_x_sphere_point(radius, centre, known_coords):
    """Find the value of the x coordinate given the coefficients of the plane"""
    y, z = known_coords
    x1 = np.sqrt(radius**2 - (y - centre[1])**2 - (z - centre[2])**2) + centre[0]
    x2 = -np.sqrt(radius**2 - (y - centre[1])**2 - (z - centre[2])**2) + centre[0]
    return x1, x2



if __name__ == "__main__":
    # load the point cloud file
    pnt_cloud_fn = 'test_data/PAE-AC-TESTS_JUL2024/METRO_OUT.txt'

    pnt_cld = np.loadtxt(pnt_cloud_fn)
    
    # flip the x axis to make the coordinate system left handed
    pnt_cld[:,1] = -1*pnt_cld[:,1]
    
    # sphere filter to get fpus
    r = 4101.1
    c = [-4101.1, 0, 0]
    rtol = 1
    sp_mask,_ = sphere_filter(pnt_cld[:,1:4], r, c, rtol)
    fltrd_pnt_cld = pnt_cld[sp_mask]
    
    # keep only the big fpu dots. These are at focal plane
    ffltrd_pnt_cld = fltrd_pnt_cld[fltrd_pnt_cld[:,7] > 2.45]
    
    # fit a sphere to the fpu dots
    r, cntrx, cntry, cntrz = sphereFit(ffltrd_pnt_cld[:,1], ffltrd_pnt_cld[:,2], ffltrd_pnt_cld[:,3])
    print(r, cntrx, cntry, cntrz)
    sphere = Sphere([cntrx, cntry, cntrz], r)
        
    # load the nominal mask posistions
    exp_mask_pos = pd.read_csv('data/ACQCAL/AC_nominal_positions.csv')
    # remove masks 18, 11, 5 (these are not fitted)
    keep_list = [5, 1, 18, 11, 8, 13]
    rm_list = [18, 11, 5]
    mask_pos = exp_mask_pos[exp_mask_pos['AC ID'].isin(keep_list)]
    
    # get the masks and store each set of points in a dictionary
    mask_points = {}
    
    # create a dataframe to store plane info
    mask_df = pd.DataFrame(columns=['AC ID', 'Nx', 'Ny', 'Nz', 'Cx', 'Cy', 'Cz', 
                                    'Ix', 'Iy', 'Iz', 'Height Diff'])
    
    # For each mask find the height difference
    for i in mask_pos['AC ID'].values:
        # select the mask
        ac = mask_pos[mask_pos['AC ID'] == i]
        
        # get the points in the mask
        # nominal centre of the mask
        mask_centre = ac[['X', 'Y', 'Z']].values[0]

        start_axis = [mask_centre[0]-5, mask_centre[1], mask_centre[2]]
        end_axis = [mask_centre[0]+5, mask_centre[1], mask_centre[2]]

        # do the filtering to get mask points
        ac_pnts = cylinder_filter(pnt_cld[:,1:4], start_axis, end_axis, radius=13)
        
        print(f'AC {i} has {ac_pnts.shape[0]} points')
        
        # fit a plane to the points
        normal, centroid = plane_fitter(ac_pnts)
        
        # define a line frrom the mask centroid along the normal
        line = Line(point=centroid, direction=normal)
        
        # find the intersection of the line with the sphere
        intersect1, intersect2 = sphere.intersect_line(line)
        
        # take the closest intersection
        if np.linalg.norm(intersect1 - centroid) < np.linalg.norm(intersect2 - centroid):
            intersect = intersect1
        else:
            intersect = intersect2
        
        # find the height difference
        height_diff = np.linalg.norm(centroid - intersect)
        print(f'AC {i} height difference: {height_diff}')
        
        # store the plane info
        row = [i, normal[0], normal[1], normal[2], centroid[0], centroid[1], 
               centroid[2], intersect[0], intersect[1], intersect[2], height_diff]
        
        mask_df.loc[len(mask_df)] = row
        # save the points
        mask_points[i] = ac_pnts

    
    # mask_df.to_csv('data/ACQCAL/mask_height_diff.csv', index=False)
    # in the region of the mask plot the mask plane and the sphere
    # define a bound around the mask to plot
    ID = 5
    plot_range = 5
    
    # load info for selected mask
    centroid = mask_df[mask_df['AC ID'] == ID][['Cx', 'Cy', 'Cz']].to_numpy()
    norm = mask_df[mask_df['AC ID'] == ID][['Nx', 'Ny', 'Nz']].to_numpy()
    intersect = mask_df[mask_df['AC ID'] == ID][['Ix', 'Iy', 'Iz']].to_numpy()
    height = mask_df[mask_df['AC ID'] == ID]['Height Diff'].to_numpy()
    
    # fine the plane equation parameters
    # Ax + By + Cz + D = 0, where [A, B, C] is the normal vector
    D = -np.dot(norm, centroid.T)
    
    # create a grid of Y, Z points in the region of interest for the mask
    Y, Z = np.meshgrid(np.linspace(centroid[0][1]-plot_range, centroid[0][1]+plot_range, 100), 
                       np.linspace(centroid[0][2]-plot_range, centroid[0][2]+plot_range, 100))
    
    # find the X value for the plane
    Xp = find_point_on_plane(norm[0][0], norm[0][1], norm[0][2], D[0], [Y, Z], missing_coord='x')
    # find the X values for the sphere. There are two solutions
    Xs = find_x_sphere_point(r, [cntrx, cntry, cntrz], [Y, Z])
    
    # get the mask points
    ac1_mask = mask_points[ID]
    
    # plot the mask plane
    fig, ax = plt.subplots(figsize=(10, 10) ,subplot_kw={'projection': '3d'})
    ax.set_title(f'AC {ID} Mask Plane and Sphere')
    ax.plot_surface(Xp, Y, Z, alpha=0.5, label='Mask Plane')
    # plot the sphere. Take the first solution
    ax.plot_surface(Xs[0], Y, Z, alpha=0.5, label='Plate Sphere')
    # ax.scatter(ac1_mask[:,1], ac1_mask[:,2], ac1_mask[:,3], label='Mask Points')
    ax.scatter(centroid[0][0], centroid[0][1], centroid[0][2], c='r', label='Mask Centroid')
    ax.scatter(intersect[0][0], intersect[0][1], intersect[0][2], c='g', label='Sphere Intersection')
    # plot the normal vector
    ax.quiver(centroid[0][0], centroid[0][1], centroid[0][2], norm[0][0], norm[0][1], 
              norm[0][2], length=height[0], label='Normal Vector')
    ax.set_aspect('equal')
    plt.legend()
    
    # show a scatter plot of all the centroids with the height difference as the colour
    fig, ax = plt.subplots()
    sct = ax.scatter(mask_df['Cy'], mask_df['Cz'], c=mask_df['Height Diff'], cmap='viridis')
    for i, txt in enumerate(mask_df['AC ID']):
        ax.annotate(int(txt), (mask_df['Cy'].iloc[i]+5, mask_df['Cz'].iloc[i]+5))
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    fig.colorbar(sct, ax=ax, label='Height Difference (mm)')
    plt.show()