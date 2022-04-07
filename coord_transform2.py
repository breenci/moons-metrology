# %%
import numpy as np
import matplotlib.pyplot as plt


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
    if norm[1]<0:
        norm = -1*norm
    
    return norm, centroid


def change_basis(v1, v2, mode='xy'):
    """Construct a change of basis matrix given two orthogonal vectors"""

    # normalise new unit vectors
    unit2 = v2/np.sqrt(np.sum(v2**2))
    unit1 = v1/np.sqrt(np.sum(v1**2))

    # find a vector perpendicular to the first two that will form a RH
    # set of orthogonal unit vectors. Different modes ensure that system is RH
    if mode == 'xy':
        unit_x = unit1
        unit_y = unit2
        unit_z = np.cross(unit_x, unit_y)

    elif mode == 'xz':
        unit_x = unit1
        unit_z = unit2
        unit_y = np.cross(unit_z, unit_x)

    elif mode == 'yz':
        unit_y = unit1
        unit_z = unit2
        unit_x = np.cross(unit_y, unit_z)

    # Construct the change of basis matrix. Columns are the coordinate vectors 
    # of the new basis vectors in the old basis. 
    A = np.vstack((unit_x, unit_y, unit_z)).T

    return A


def get_translate_mat(V_translation):
    """Construct a matrix which translates by input vector"""

    # 3D translation matrix is a 4 X 4 identity matrix with the translation
    # vector in the first 3 rows of final column
    t_mat = np.eye(4)
    t_mat[:3,3] = -1*V_translation

    return t_mat


def get_pln_t_mat(pln_pnts, new_origin, y0):
    """Construct the transformation matrix for a transformation into new basis
    with x defined by the normal of a plane and y defined by a point"""
    
    # fit a plane and find the unit normal
    new_x,_ = plane_fitter(pln_pnts)

    # Define the direction of the y basis vector.
    # y_dir will lie in the xy plane
    y_dir = y0 - new_origin

    # Z vector is perpendicular to the xy plane
    new_z = np.cross(new_x, y_dir)

    #get change of basis (COB) and translation matrices
    cob_mat = change_basis(new_x, new_z, mode='xz')
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
    """Do the matrix transformation"""

    # pad input coords with ones to allow translation
    pad_coords = np.vstack((coords.T, np.ones_like(coords[:,0])))

    # do transformation
    pad_trans_coords = np.matmul(trans_mat, pad_coords)

    # unpad
    trans_coords = pad_trans_coords[:3,:].T

    return trans_coords



def do_transform(data, plane_inds, origin_ind, y0_ind):
    pln_pnts = data[plane_inds]
    origin = data[origin_ind]
    y0 = data[y0_ind]

    trans_mat = get_pln_t_mat(pln_pnts, origin, y0)

    transformed = matrix_transform(data, trans_mat)

    return transformed


if __name__ == '__main__':

    ten_deg = np.loadtxt('caltest_25_03_22/10_deg.txt')
    coords = ten_deg[:,1:4]

    pln_inds = np.arange(1,17)
    orgn_ind = 8
    y0_ind = 7

    transformed = do_transform(coords, pln_inds, orgn_ind, y0_ind)


    plt.scatter(transformed[:,1], transformed[:,2])
    plt.show()


    
# %%
