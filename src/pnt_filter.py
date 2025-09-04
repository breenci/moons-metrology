"""
Module to filter spurious points from a metrology ouput point cloud
"""

import numpy as np
import matplotlib.pyplot as plt


def sphere_filter(coords, r, c, tol):
    """
    Filters out any points not on a spherical surface to within a tolerance

    :param coords: 3D point cloud coordinates with X, Y, and Z in columns
    :type data: N x 3 numpy array
    :param r: Radius of sphere which defines filtering surface
    :type r: float
    :param c: Centre of sphere which defines filtering surface
    :type c: 1 x 3 numpy array
    :param tol: values within this tolerance are accepted
    :type tol: float
    :return: Filtered 3D point cloud coordinates with X, Y, and Z in columns
    :rtype: N x 3 numpy array
    """

    # Subtract centre of sphere from coords to satisfy equation of sphere
    cntrd_coords = coords - c

    # Calculate how far each point is from sphere surface
    dist = np.sqrt(np.sum(cntrd_coords**2, axis=1))
    diff = abs(dist - r)

    # Reject points not on surface to within tol
    sphere_mask = diff < tol
    out_coords = coords[sphere_mask]

    return sphere_mask, out_coords


def size_filter(coords, sizes, size_arr, tol):
    """
    Filters out points not within a tolerance of given sizes

    :param coords: 3D point cloud coordinates with X, Y, and Z in columns
    :type data: N x 3 numpy array
    :param sizes: Array of point sizes
    :type sizes: 1d numpy array
    :param size_arr: Array of desired sizes
    :type size_arr: 1d numpy array
    :param tol: tolerance
    :type tol: float
    :return: Filtered data
    :rtype: N x 3 numpy array
    """

    # Define upper and lower bounds
    upper_bnds = size_arr + tol
    lower_bnds = size_arr - tol
    
    # raise an error is bnds overlap
    diff = np.diff(size_arr)
    if np.any(diff < 2*tol):
        raise ValueError('Bounds overlap')
    
    # Create logical mask for first set of bounds then loop through the other
    # bounds and update the mask with each loop
    in_bnds_mask = (sizes < upper_bnds[0]) & (sizes > lower_bnds[0])
    # Make sizes discrete and match to known values
    sizes[in_bnds_mask] = size_arr[0]

    for i in range(1, len(upper_bnds)):
        cond_mask = (sizes < upper_bnds[i]) & (sizes > lower_bnds[i])
        sizes[cond_mask] = size_arr[i]
        in_bnds_mask = in_bnds_mask | cond_mask

    # Filter dataset using mask
    data_out = coords[in_bnds_mask]
    sizes_class = sizes[in_bnds_mask]

    return in_bnds_mask, data_out, sizes_class


def cylinder_filter(points, cylinder_axis_start, cylinder_axis_end, radius):
    """
    Filters a 3D point cloud to retain only points inside a cylinder.

    :param points: 3D point cloud coordinates with X, Y, and Z in columns
    :type points: numpy.ndarray
    :param cylinder_axis_start: Start point of the cylinder axis
    :type cylinder_axis_start: array-like
    :param cylinder_axis_end: End point of the cylinder axis
    :type cylinder_axis_end: array-like
    :param radius: Radius of the cylinder
    :type radius: float
    :return: Filtered 3D point cloud coordinates with X, Y, and Z in columns
    :rtype: numpy.ndarray
    """
    # Convert inputs to numpy arrays
    p0 = np.array(cylinder_axis_start)
    p1 = np.array(cylinder_axis_end)
    v = p1 - p0  # Cylinder axis vector
    v_length = np.linalg.norm(v)
    v_unit = v / v_length

    filtered_points = []

    for point in points:
        # Vector from p0 to the point
        w = point - p0
        # Project w onto the cylinder axis
        projection_length = np.dot(w, v_unit)
        
        # Check if the projection falls within the cylinder height
        if 0 <= projection_length <= v_length:
            # Compute the perpendicular distance from the point to the axis
            closest_point_on_axis = p0 + projection_length * v_unit
            distance_to_axis = np.linalg.norm(point - closest_point_on_axis)
            
            if distance_to_axis <= radius:
                filtered_points.append(point)

    return np.array(filtered_points)


def points_in_box(coordinates, box_point, box_lengths):
    """
    Find indices of points within a 3D box defined by a point and lengths.

    Parameters:
    - coordinates: List of 3D coordinates (list of lists or numpy array).
    - box_point: Reference point of the box (list or tuple of three values).
    - box_lengths: Lengths of the box along each dimension (list or tuple of three values).

    Returns:
    - List of indices of points within the specified 3D box.
    """
    indices = []

    for i, coord in enumerate(coordinates):
        x_in_range = box_point[0] <= coord[0] <= box_point[0] + box_lengths[0]
        y_in_range = box_point[1] <= coord[1] <= box_point[1] + box_lengths[1]
        z_in_range = box_point[2] <= coord[2] <= box_point[2] + box_lengths[2]

        if x_in_range and y_in_range and z_in_range:
            indices.append(i)

    return indices