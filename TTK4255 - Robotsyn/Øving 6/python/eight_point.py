import numpy as np
from normalize_points import *

def eight_point(uv1, uv2):
    """ Given n >= 8 point matches, (u1 v1) <-> (u2 v2), compute the
    fundamental matrix F that satisfies the equations

        (u2 v2 1)^T * F * (u1 v1 1) = 0

    Args:
        uv1: (n x 2 array) Pixel coordinates in image 1.
        uv2: (n x 2 array) Pixel coordinates in image 2.

    Returns:
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1
             to lines in image 2.

    See HZ Ch. 11.2: The normalized 8-point algorithm (p.281).
    """

    # todo: Compute F
    uv1_norm, T1 = normalize_points(uv1)
    uv2_norm, T2 = normalize_points(uv2)

    A_col1 = np.array([uv2_norm[:,0] * uv1_norm[:,0]]).T
    A_col2 = np.array([uv2_norm[:,0] * uv1_norm[:,1]]).T
    A_col3 = np.array([uv2_norm[:,0]]).T
    A_col4 = np.array([uv2_norm[:,1] * uv1_norm[:,0]]).T
    A_col5 = np.array([uv2_norm[:,1] * uv1_norm[:,1]]).T
    A_col6 = np.array([uv2_norm[:,1]]).T
    A_col7 = np.array([uv1_norm[:,0]]).T
    A_col8 = np.array([uv1_norm[:,1]]).T
    A_col9 = np.ones((uv1.shape[0],1))

    A = np.concatenate((A_col1, A_col2, A_col3, A_col4, A_col5, A_col6, A_col7, A_col8, A_col9), axis=1)
    _,_,vh = np.linalg.svd(A)

    f = vh[8,:]
    F = np.reshape(f, (3,3))
    F = closest_fundamental_matrix(F)

    F = T1.T @ F @ T1
    return F

def closest_fundamental_matrix(F):
    """
    Computes the closest fundamental matrix in the sense of the
    Frobenius norm. See HZ, Ch. 11.1.1 (p.280).
    """

    # todo: Compute the correct F
    u,d,vh = np.linalg.svd(F)
    d[2] = 0
    d = np.diag(d)

    F_new = u @ d @ vh

    return F_new
