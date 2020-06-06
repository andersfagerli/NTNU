import matplotlib.pyplot as plt
import numpy as np
from common import draw_frame

def estimate_H(xy, XY):
    #
    # Task 2: Implement estimate_H
    #
    rows,_= XY.shape
    A = np.zeros((2*rows,9))
    for i in range(len(XY)):
        Xi = XY[i,0]
        Yi = XY[i,1]
        xi = xy[i,0]
        yi = xy[i,1]

        j = i*2
        A[j:(j+2),:] = np.array([[Xi, Yi, 1, 0, 0, 0, -Xi*xi, -Yi*xi, -xi],
                                 [0, 0, 0, Xi, Yi, 1, -Xi*yi, -Yi*yi, -yi]])
    _,_,vh = np.linalg.svd(A)
    _,columns = vh.T.shape
    h = vh.T[:,columns-1]
    H = np.reshape(h,(3,3))
    return H

def decompose_H(H):
    #
    # Task 3a: Implement decompose_H
    #
    positive_scale = np.linalg.norm(H[:,0])
    negative_scale = -1*positive_scale

    t1 = H[:,2] / positive_scale
    t2 = H[:,2] / negative_scale


    r1_positive = H[:,0] / positive_scale
    r1_negative = H[:,0] / negative_scale

    r2_positive = H[:,1] / positive_scale
    r2_negative = H[:,1] / negative_scale

    r3_positive = np.cross(r1_positive, r2_positive)
    r3_negative = np.cross(r1_negative, r2_negative)

    # Transposing into column vectors
    t1 = np.array([t1]).T
    t2 = np.array([t2]).T
    r1_positive = np.array([r1_positive]).T
    r1_negative = np.array([r1_negative]).T
    r2_positive = np.array([r2_positive]).T
    r2_negative = np.array([r2_negative]).T
    r3_positive = np.array([r3_positive]).T
    r3_negative = np.array([r3_negative]).T

    R1 = np.column_stack((r1_positive, r2_positive, r3_positive))
    R2 = np.column_stack((r1_negative, r2_negative, r3_negative))

    T1 = np.block([
        [R1,            t1],
        [np.zeros((1,3)), 1]
    ])

    T2 = np.block([
        [R2,            t2],
        [np.zeros((1,3)), 1]
    ])

    print(T1)
    print(T2)
    
    return T1, T2

def choose_solution(T1, T2):
    #
    # Task 3b: Implement choose_solution
    #
    if (T1[2,3] >= 0):
        return T1
    else:
        return T2

K           = np.loadtxt('../data/cameraK.txt')
all_markers = np.loadtxt('../data/markers.txt')
XY          = np.loadtxt('../data/model.txt')
n           = len(XY)


for image_number in range(23):
    I = plt.imread('../data/video%04d.jpg' % image_number)
    markers = all_markers[image_number,:]
    markers = np.reshape(markers, [n, 3])
    matched = markers[:,0].astype(bool) # First column is 1 if marker was detected
    uv = markers[matched, 1:3] # Get markers for which matched = 1

    # Convert pixel coordinates to normalized image coordinates
    xy = (uv - K[0:2,2])/np.array([K[0,0], K[1,1]])

    H = estimate_H(xy, XY[matched, :2])
    T1,T2 = decompose_H(H)
    T = choose_solution(T1, T2)

    # Compute predicted corner locations using model and homography
    uv_hat = (K@H@XY.T)
    uv_hat = (uv_hat/uv_hat[2,:]).T

    plt.clf()
    plt.imshow(I, interpolation='bilinear')
    draw_frame(K, T, scale=7)
    plt.scatter(uv[:,0], uv[:,1], color='red', label='Observed')
    plt.scatter(uv_hat[:,0], uv_hat[:,1], marker='+', color='yellow', label='Predicted')
    plt.legend()
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.savefig('../data/out%04d.png' % image_number)
