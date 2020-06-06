import sys
sys.path.append("../pointcloud")

# External libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# Own libraries
import SIFT as sift
import util

### Constants ###
DRAW = True
OFFSET = 0.05

### Read from files ###
img1 = cv2.imread('data/out001.png',0)  # queryImage
img2 = cv2.imread('data/out002.png',0)  # trainImage

K1 = np.loadtxt('oving6/K1.txt')
K2 = np.loadtxt('oving6/K2.txt')

### Generate keypoints and descriptors ###
mask = np.zeros(img1.shape[:2], dtype=np.uint8)
print(img1.shape)
start_y = int(img1.shape[0]/10)
start_x = int(img1.shape[1]/4)
cv2.rectangle(mask, (start_x, start_y), (img1.shape[1] - start_x, img1.shape[0] - start_y), (255), thickness=-1)

sift_kp1, sift_des1 = sift.keypointAndDescriptor(img1, nfeatures=10000, mask=mask)
sift_kp2, sift_des2 = sift.keypointAndDescriptor(img2, nfeatures=10000, mask=mask)

### Generate matches between images ###
sift_pts1, sift_pts2, sift_matches, sift_matches_mask = sift.flannMatches2D2D(sift_kp1, sift_kp2, sift_des1, sift_des2)

if DRAW:
    sift.drawMatches(img1, img2, sift_kp1, sift_kp2, sift_matches, sift_matches_mask)

### Get fundamental matrix ###
sift_pts1 = np.int32(sift_pts1)
sift_pts2 = np.int32(sift_pts2)

F, mask = cv2.findFundamentalMat(sift_pts1, sift_pts2, cv2.FM_RANSAC, ransacReprojThreshold=0.05, confidence=0.99999)

# Choose only inlier points from fundemental mask 
sift_pts1 = sift_pts1[mask.ravel()==1]
sift_pts2 = sift_pts2[mask.ravel()==1]

### Get camera projection matrices P=K[R|t] ###
H, mask = cv2.findHomography(sift_pts1, sift_pts2, method=0)
n, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K1)

Rts = [(Rs[0],np.squeeze(Ts[0].T)), (Rs[1],np.squeeze(Ts[1].T)), (Rs[2],np.squeeze(Ts[2].T)), (Rs[3],np.squeeze(Ts[3].T))]
R,t = util.choose_solution(sift_pts1, sift_pts2, K1, K2, Rts)
P1,P2 = util.camera_matrices(K1, K2, R, t)

uv1 = sift_pts1
uv2 = sift_pts2

# ### Epipolar matching ###
uv1 = np.array([sift_kp1[i].pt for i in range(len(sift_kp1))])
uv2 = util.epipolar_match(img1, img2, F, uv1)

### Triangulate to generate 3D points ###
n = len(uv1)
X = np.array([util.linear_triangulation(uv1[i], uv2[i], P1, P2) \
    for i in range(n)])

### Display pointcloud ###
util.show_point_cloud(X[:,:3],
     xlim=[min(X[:,0])-OFFSET,max(X[:,0])+OFFSET],
     ylim=[min(X[:,1])-OFFSET,max(X[:,1])+OFFSET],
     zlim=[min(X[:,2])-OFFSET,max(X[:,2])+OFFSET])
