"""
Module for doing coordinate transformations for metrology measurements
"""
import numpy as np


def plane_fitter(point_coords):
    """Fit a plane to a set of points and return the unit normal.

    :param point_coords: N x 3 array of point coordinates
    :type point_coords: numpy.ndarray
    :return: Unit normal vector of the fitted plane and the centroid of the points
    :rtype: tuple
    """

    # Subtract the centroid from the set of points
    centroid = np.mean(point_coords, axis=0)
    cntrd_pnts = point_coords - centroid
    
    # Perform singular value decomposition
    svd, _, _ = np.linalg.svd(cntrd_pnts.T)
    
    # Final column of the svd gives the unit normal
    norm = svd[:,2]
    
    # Take the +ve norm
    if norm[1]<0:
        norm = -1*norm
    
    return norm, centroid


def change_basis(v1, v2, mode='xy'):
    """Construct a change of basis matrix given two orthogonal vectors

    :param v1: First orthogonal vector
    :type v1: numpy.ndarray
    :param v2: Second orthogonal vector
    :type v2: numpy.ndarray
    :param mode: Specifies orientation of basis. e.g 'xy' means v1 is x and v2 is y
    :type mode: str, optional
    :return: Change of basis matrix
    :rtype: numpy.ndarray
    """

    # normalise new unit vectors
    unit2 = v2/np.sqrt(np.sum(v2**2))
    unit1 = v1/np.sqrt(np.sum(v1**2))

    # find a vector perpendicular to the first two that will form a LH
    # set of orthogonal unit vectors. Different modes ensure that system is LH
    if mode == 'xy':
        unit_x = unit1
        unit_y = unit2
        unit_z = np.cross(unit_y, unit_x)

    elif mode == 'xz':
        unit_x = unit1
        unit_z = unit2
        unit_y = np.cross(unit_x, unit_z)

    elif mode == 'yz':
        unit_y = unit1
        unit_z = unit2
        unit_x = np.cross(unit_z, unit_y)

    # Construct the change of basis matrix. Columns are the coordinate vectors 
    # of the new basis vectors in the old basis. 
    A = np.vstack((unit_x, unit_y, unit_z)).T

    return A


def get_translate_mat(V_translation):
    """Construct a matrix which translates by input vector
    
    :param V_translation: 3D translation vector
    :type V_translation: numpy.ndarray
    :return: 4x4 translation matrix
    :rtype: numpy.ndarray
    """

    # 3D translation matrix is a 4 X 4 identity matrix with the translation
    # vector in the first 3 rows of final column
    t_mat = np.eye(4)
    t_mat[:3,3] = -1*V_translation

    return t_mat


def get_pln_t_mat(pln_pnts, new_origin, z0):
    """Construct the transformation matrix for a transformation into new basis
    with x defined by the normal of a plane and y defined by a point
    
    :param pln_pnts: Points that define the plane
    :type pln_pnts: numpy.ndarray
    :param new_origin: New origin point
    :type new_origin: numpy.ndarray
    :param z0: Point defining the z direction
    :type z0: numpy.ndarray
    :return: Transformation matrix
    :rtype: numpy.ndarray
    """
    
    # fit a plane and find the unit normal
    new_x,_ = plane_fitter(pln_pnts)

    # Define the direction of the y basis vector.
    # z_dir will lie in the xz plane
    z_dir = z0 - new_origin

    # y vector is perpendicular to the xz plane
    new_y = np.cross(new_x, z_dir)

    #get change of basis (COB) and translation matrices
    cob_mat = change_basis(new_x, new_y, mode='xy')
    t_mat = get_translate_mat(new_origin)
    hom_cob_mat = np.eye(4)

    # Note: COB matrix M defined as: old coords = M * new coords
    # -> M_inv * old_coords = M_transpose * old coords = new coords
    # This is why we take the transpose here
    hom_cob_mat[:3,:3] = cob_mat.T

    # combine into one matrix that will define translation and COB
    trans_mat = np.matmul(hom_cob_mat, t_mat)
    
    return trans_mat


def matrix_transform(coords, trans_mat):
    """Do the matrix transformation

    :param coords: Input coordinates
    :type coords: numpy.ndarray
    :param trans_mat: Transformation matrix
    :type trans_mat: numpy.ndarray
    :return: Transformed coordinates
    :rtype: numpy.ndarray
    """

    # pad input coords with ones to allow translation
    pad_coords = np.vstack((coords.T, np.ones_like(coords[:,0])))

    # do transformation
    pad_trans_coords = np.matmul(trans_mat, pad_coords)

    # unpad
    trans_coords = pad_trans_coords[:3,:].T

    return trans_coords


def project_point_to_plane(normal, point, pln_pnt=(0,0,0)):
    """Projects a point to a plane

    :param normal: Normal vector of the plane
    :type normal: numpy.ndarray
    :param point: Point to project
    :type point: numpy.ndarray
    :param pln_pnt: A point on the plane
    :type pln_pnt: tuple
    :return: Projected point
    :rtype: numpy.ndarray
    """

    #normalise vector
    normal = normal/np.sqrt(np.sum(normal**2))

    pln_to_pnt = point - pln_pnt
    
    # project to plane
    new_point = point - normal*np.dot(normal, pln_to_pnt)
    
    return new_point


def do_transform(data, plane_inds, origin_ind, y0_ind, origin_mode='ind'):
    """Do a transformation of data into a new basis defined by a plane and a point
    with x defined by the normal of the plane and z defined by the point

    :param data: Input data points
    :type data: numpy.ndarray
    :param plane_inds: Indices of points defining the plane
    :type plane_inds: list
    :param origin_ind: Index of the origin point
    :type origin_ind: int
    :param y0_ind: Index of the point defining the y direction
    :type y0_ind: int
    :param origin_mode: Mode for defining the origin point
    :type origin_mode: str
    :return: Transformed data points
    :rtype: numpy.ndarray
    """
    # get the points defining the plane
    pln_pnts = data[plane_inds]

    # get the point defining the z direction
    y0 = data[y0_ind]

    # get the point defining the origin either as an index or a point
    if origin_mode == 'pnt':
        origin = origin_ind
        
    if origin_mode == 'ind':
        origin = data[origin_ind]

    # get the transformation matrix
    trans_mat = get_pln_t_mat(pln_pnts, origin, y0)

    # apply the transformation
    transformed = matrix_transform(data, trans_mat)

    return transformed