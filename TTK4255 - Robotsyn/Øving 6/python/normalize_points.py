import numpy as np

def normalize_points(pts):
    """ Computes a normalizing transformation of the points such that
    the points are centered at the origin and their mean distance from
    the origin is equal to sqrt(2).

    See HZ, Ch. 4.4.4: Normalizing transformations (p107).

    Args:
        pts:    Input 2D point array of shape n x 2

    Returns:
        pts_n:  Normalized 2D point array of shape n x 2
        T:      The normalizing transformation in 3x3 matrix form, such
                that for a point (x,y), the normalized point (x',y') is
                found by multiplying T with the point:

                    |x'|       |x|
                    |y'| = T * |y|
                    |1 |       |1|
    """

    # Todo: Compute pts_n and T
    rows, cols = pts.shape

    mu = np.array([np.mean(pts[:,0]), np.mean(pts[:,1])])
    sigma = np.mean(np.linalg.norm(pts-mu, axis = 1))
    
    T = np.array([[np.sqrt(2)/sigma,                0, -np.sqrt(2)/sigma * mu[0]],
                  [               0, np.sqrt(2)/sigma, -np.sqrt(2)/sigma * mu[1]],
                  [               0,                0,                          1]])

    pts_n = np.zeros((rows, cols))
    pts_homogenous = np.concatenate((pts, np.ones((rows,1))), axis=1)
    
    for i in range(rows):
        xi = np.array([pts_homogenous[i,:]]).T
        xi_hat = T @ xi
        pts_n[i,:] = np.array(xi_hat[0:2].T)


    return pts_n, T
