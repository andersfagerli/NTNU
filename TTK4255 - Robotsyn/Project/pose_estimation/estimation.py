import numpy as np
# TODO:
# 1. Generate initial P with DLT. Run inside RANSAC loop to find inliers
# 2.  

# Should choose a subset of correspondences and compute P with DLT. Then compare reprojection error of 3d to 2d with remaining 3d points.
def RANSAC():
    

def DLT(points2D, points3D):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]])