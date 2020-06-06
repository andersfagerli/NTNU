import numpy as np
import matplotlib.pyplot as plt
import cv2

def camera_matrices(K1, K2, R, t):
    """ Computes the projection matrix for camera 1 and camera 2.

    Args:
        K1,K2: Intrinsic matrix for camera 1 and camera 2.
        R,t: The rotation and translation mapping points in camera 1 to points in camera 2.

    Returns:
        P1,P2: The projection matrices with shape 3x4.
    """

    P1 = K1@np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2 = K2@np.column_stack((R, t))
    return P1, P2

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

def choose_solution(uv1, uv2, K1, K2, Rts):
    """
    Chooses among the rotation and translation solutions Rts
    the one which gives the most points in front of both cameras.
    """

    n = len(uv1)
    best = (0, 0)
    for i, (R,t) in enumerate(Rts):
        P1,P2 = camera_matrices(K1, K2, R, t)
        X1 = np.array([linear_triangulation(uv1[j], uv2[j], P1, P2) for j in range(n)])
        X2 = X1 @ R.T + t
        visible = np.logical_and(X1[:,2] > 0, X2[:,2] > 0)
        num_visible = np.sum(visible)
        if num_visible > best[1]:
            best = (i, num_visible)
    print('Choosing solution %d (%d points visible)' % (best[0], best[1]))
    return Rts[best[0]]


def essential_from_fundamental(F, K1, K2):
    """ Computes the essential matrix from F.

    See HZ Ch. 9.6 (p257): Formula 9.12.

    Args:
        F:  Fundamental Matrix
        K1: Intrinsic matrix for camera 1
        K2: Intrinsic matrix for camera 2

    Returns:
        E:  Essential Matrix
    """

    return K1.T @ F @ K2


def motion_from_essential(E):
    """ Computes the four possible decompositions of E into
    a relative rotation and translation.

    See HZ Ch. 9.7 (p259): Result 9.19
    """
    U, s, VT = np.linalg.svd(E)

    # Make sure we return rotation matrices with det(R) == 1
    if np.linalg.det(U) < 0: U = -U
    if np.linalg.det(VT) < 0: VT = -VT

    W = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U@W@VT
    R2 = U@W.T@VT
    t1 = U[:,2]
    t2 = -U[:,2]
    return [(R1,t1), (R1,t2), (R2, t1), (R2, t2)]

def epipolar_match(I1, I2, F, uv1):
    """
    For each point in uv1, finds the matching point in image 2 by
    an epipolar line search.

    Args:
        I1:  (H x W matrix) Grayscale image 1
        I2:  (H x W matrix) Grayscale image 2
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1 to lines in image 2
        uv1: (n x 2 array) Points in image 1

    Returns:
        uv2: (n x 2 array) Best matching points in image 2.
    """
    w = 10
    uv2 = np.zeros(uv1.shape)
    for i, (u1,v1) in enumerate(uv1):
        if u1 < w or v1 < w or u1 > I1.shape[1]-w or v1 > I1.shape[0]-w:
            continue
        l = F@np.array((u1,v1,1))
        W1 = I1[int(v1)-w:int(v1)+w+1, int(u1)-w:int(u1)+w+1]
        best_err = np.inf
        best_u2 = w
        for u2 in range(w, I2.shape[1]-w):
            v2 = int(round(-(l[2] + u2*l[0])/l[1]))
            if v2 < w or v2 >= I2.shape[0]-w:
                continue
            W2 = I2[v2-w:v2+w+1, u2-w:u2+w+1]
            err = np.sum(np.absolute(W1 - W2))
            if err < best_err:
                best_err = err
                best_u2 = u2

        uv2[i,0] = best_u2
        uv2[i,1] = -(l[2] + best_u2*l[0])/l[1]
    return uv2


def linear_triangulation(uv1, uv2, P1, P2):
    """
    Compute the 3D position of a single point from 2D correspondences.

    Args:
        uv1:    2D projection of point in image 1.
        uv2:    2D projection of point in image 2.
        P1:     Projection matrix with shape 3 x 4 for image 1.
        P2:     Projection matrix with shape 3 x 4 for image 2.

    Returns:
        X:      3D coordinates of point in the camera frame of image 1.
                (not homogeneous!)

    See HZ Ch. 12.2: Linear triangulation methods (p312)
    """

    A = np.empty((4,4))
    A[0,:] = uv1[0]*P1[2,:] - P1[0,:]
    A[1,:] = uv1[1]*P1[2,:] - P1[1,:]
    A[2,:] = uv2[0]*P2[2,:] - P2[0,:]
    A[3,:] = uv2[1]*P2[2,:] - P2[1,:]
    U,s,VT = np.linalg.svd(A)
    X = VT[3,:]
    return X[:3]/X[3]

def getTransformation(R, t):
    T = np.concatenate((R, t), axis=1)
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)

    return T

def draw_line(l, **args):
    """
    Draws the line satisfies the line equation
        x l[0] + y l[1] + l[2] = 0
    clipped to the current plot's box (xlim, ylim).
    """

    def clamp(a, b, a_min, a_max, A, B, C):
        if a < a_min or a > a_max:
            a = np.fmax(a_min, np.fmin(a_max, a))
            b = -(C + a*A)/B
        return a, b

    x_min,x_max = np.sort(plt.xlim())
    y_min,y_max = np.sort(plt.ylim())
    if abs(l[1]) > abs(l[0]):
        x1 = x_min
        x2 = x_max
        y1 = -(l[2] + x1*l[0])/l[1]
        y2 = -(l[2] + x2*l[0])/l[1]
        y1,x1 = clamp(y1, x1, y_min, y_max, l[1], l[0], l[2])
        y2,x2 = clamp(y2, x2, y_min, y_max, l[1], l[0], l[2])
    else:
        y1 = y_min
        y2 = y_max
        x1 = -(l[2] + y1*l[1])/l[0]
        x2 = -(l[2] + y2*l[1])/l[0]
        x1,y1 = clamp(x1, y1, x_min, x_max, l[0], l[1], l[2])
        x2,y2 = clamp(x2, y2, x_min, x_max, l[0], l[1], l[2])
    plt.plot([x1, x2], [y1, y2], **args)

def rgb2gray(I):
    """
    Converts a red-green-blue (RGB) image to grayscale brightness.
    """
    return 0.2989*I[:,:,0] + 0.5870*I[:,:,1] + 0.1140*I[:,:,2]

def show_point_matches(I1, I2, uv1, uv2, F=None):
    """
    Plots k randomly chosen matching point pairs in image 1 and
    image 2. If the fundamental matrix F is given, it also plots the
    epipolar lines.
    """

    k = 10
    sample = np.random.choice(range(len(uv1)), size=k, replace=False)
    uv1 = uv1[sample,:]
    uv2 = uv2[sample,:]

    plt.figure(figsize=(6,4))
    colors = plt.cm.get_cmap('Set1', k).colors
    plt.subplot(121)
    plt.imshow(I1)
    plt.scatter(uv1[:,0], uv1[:,1], s=100, marker='x', c=colors)
    plt.subplot(122)
    plt.imshow(I2)
    plt.scatter(uv2[:,0], uv2[:,1], s=100, marker='o', zorder=10, facecolor='none', edgecolors=colors, linewidths=2)
    if not F is None:
        for i,(u1,v1) in enumerate(uv1):
            l = F@np.array((u1,v1,1))
            draw_line(l, linewidth='1', color=colors[i])
    plt.tight_layout()

def show_point_cloud(X, xlim, ylim, zlim):
    """
    Creates a mouse-controllable 3D plot of the input points.
    """
    plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')

    # This could be changed to use scatter if you want to
    # provide a per-point color. Otherwise, the plot function
    # is much faster.
    ax.plot(X[:,0], X[:,2], X[:,1], '.')

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('x')
    ax.set_zlabel('y')
    ax.set_ylabel('z')
    plt.show()

def draw_frame(ax, T, scale = 0.1):
    """
    Draws a coordinate-frame with translation/rotation T from origin
    Args:
        ax:     3D pyplot to plot frame in
        T:      4x4 homogeneous transformation [R|t]
        scale:  Size of frame
    """
    center = T @ np.array([0,0,0,1])
    x = T @ np.array([scale,0,0,1])
    z = T @ np.array([0,scale,0,1])
    y = T @ np.array([0,0,scale,1])

    ax.plot([center[0], x[0]], [center[1], x[1]],[center[2], x[2]], color='#cc4422')
    ax.plot([center[0], y[0]], [center[1], y[1]],[center[2], y[2]], color='#11ff33')
    ax.plot([center[0], z[0]], [center[1], z[1]],[center[2], z[2]], color='#3366ff')

def draw_origin(ax, scale):
    T = np.eye(4)
    center = T @ np.array([0,0,0,1])
    x = T @ np.array([scale,0,0,1])
    z = T @ np.array([0,scale,0,1])
    y = T @ np.array([0,0,scale,1])

    ax.plot([center[0], x[0]], [center[1], x[1]],[center[2], x[2]], color='grey', linestyle = "dashed")
    ax.plot([center[0], y[0]], [center[1], y[1]],[center[2], y[2]], color='grey', linestyle = "dashed")
    ax.plot([center[0], z[0]], [center[1], z[1]],[center[2], z[2]], color='grey', linestyle = "dashed")