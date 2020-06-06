import sys
sys.path.append("../pointcloud")

# External libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# Own libraries
import SIFT as sift
from read_write_model import read_model
import util

### Read from files ###
img1 = cv2.imread('oving6/im1.png',0)  # queryImage
img2 = cv2.imread('oving6/im2.png',0)  # trainImage

K1 = np.loadtxt('oving6/K1.txt')
K2 = np.loadtxt('oving6/K2.txt')

### Generate keypoints and descriptors ###
sift_kp1, sift_des1 = sift.keypointAndDescriptor(img1, nfeatures=1500)
sift_kp2, sift_des2 = sift.keypointAndDescriptor(img2, nfeatures=1500)

### Generate matches between images ###
sift_pts1, sift_pts2, sift_matches, sift_matches_mask = sift.flannMatches2D2D(sift_kp1, sift_kp2, sift_des1, sift_des2)

sift.drawMatches(img1, img2, sift_kp1, sift_kp2, sift_matches, sift_matches_mask)

### Get fundamental matrix ###
sift_pts1 = np.int32(sift_pts1)
sift_pts2 = np.int32(sift_pts2)

F, mask = cv2.findFundamentalMat(sift_pts1, sift_pts2, cv2.FM_RANSAC, ransacReprojThreshold=0.05, confidence=0.99999)

# Choose only inlier points from fundemental mask 
sift_pts1 = sift_pts1[mask.ravel()==1]
sift_pts2 = sift_pts2[mask.ravel()==1]

uv1 = sift_pts1
uv2 = sift_pts2

### Get camera projection matrices P=K[R|t] ###
H, mask = cv2.findHomography(sift_pts1, sift_pts2, method=0)
n, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K1)

Rts = [(Rs[0],np.squeeze(Ts[0].T)), (Rs[1],np.squeeze(Ts[1].T)), (Rs[2],np.squeeze(Ts[2].T)), (Rs[3],np.squeeze(Ts[3].T))]
R,t = util.choose_solution(sift_pts1, sift_pts2, K1, K2, Rts)
P1,P2 = util.camera_matrices(K1, K2, R, t)

### Epipolar matching ###
uv1 = np.array([sift_kp1[i].pt for i in range(len(sift_kp1))])
uv2 = util.epipolar_match(img1, img2, F, uv1)

### Triangulate to generate 3D points ###
n = len(uv1)
X = np.array([util.linear_triangulation(uv1[i], uv2[i], P1, P2) \
    for i in range(n)])

pointcloud = (X, sift_des1)
print(X.shape)
print(sift_des1.shape)
### 3D to 2D matching ###
pts_img, pts3D, _, _ = sift.flannMatches3D2D(sift_kp2, sift_des2, pointcloud[0], pointcloud[1])

### Pose estimation ###
#success, rot_vec, translation, inliers = cv2.solvePnPRansac(pts3D, pts_img, K2, None) # Using (X, uv2, K2, none) works better tho ..?
success, rot_vec, translation = cv2.solvePnP(pts3D, pts_img, K2, None)

# Shuffle terms so they coincide with our coordinate system
rot_vec = np.array([rot_vec[0], rot_vec[2], rot_vec[1]])
R,_ = cv2.Rodrigues(rot_vec)

translation = np.array([-translation[0], translation[2], -translation[1]])

### Conversion to transformation matrix T = [R|t] ###
T = util.getTransformation(R, translation)

### Plotting ###
OFFSET = 0.5 + 3
xlim = [min(X[:,0])-OFFSET,max(X[:,0])+OFFSET]
ylim = [min(X[:,1])-OFFSET,max(X[:,1])+OFFSET]
zlim = [min(X[:,2])-OFFSET,max(X[:,2])+OFFSET]

plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

ax.plot(X[:,0], X[:,2], X[:,1], '.', markersize=1)
util.draw_frame(ax, np.eye(4))
util.draw_frame(ax, T)

ax.set_xlim(xlim)
ax.set_ylim(zlim)
ax.set_zlim([ylim[1], ylim[0]])
ax.set_xlabel('x')
ax.set_zlabel('y')
ax.set_ylabel('z')
plt.show()