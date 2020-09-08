import sys
sys.path.append("../pointcloud")

# External libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# Own libraries
import ORB as orb
import util

### Constants ###
DRAW = True
OFFSET = 0.05

### Read from files ###
img1 = cv2.imread('oving6/im1.png',0)  # queryImage
img2 = cv2.imread('oving6/im2.png',0)  # trainImage

K1 = np.loadtxt('oving6/K1.txt')
K2 = np.loadtxt('oving6/K2.txt')

### Generate keypoints and descriptors ###
orb_kp1, orb_des1 = orb.keypointAndDescriptor(img1, nfeatures=1500)
orb_kp2, orb_des2 = orb.keypointAndDescriptor(img2, nfeatures=1500)

### Generate matches between images ###
orb_pts1, orb_pts2, orb_matches, orb_matches_mask = orb.flannMatches(orb_kp1, orb_kp2, orb_des1, orb_des2)

if DRAW:
    orb.drawMatches(img1, img2, orb_kp1, orb_kp2, orb_matches, orb_matches_mask)

### Get fundamental matrix ###
F, mask = cv2.findFundamentalMat(orb_pts1, orb_pts2, cv2.FM_RANSAC, ransacReprojThreshold=0.05, confidence=0.99999)

# Choose only inlier points from fundemental mask 
orb_pts1 = orb_pts1[mask.ravel()==1]
orb_pts2 = orb_pts2[mask.ravel()==1]

### Get camera projection matrices P=K[R|t] ###
H, mask = cv2.findHomography(orb_pts1, orb_pts2, method=0)
n, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K1)

Rts = [(Rs[0],np.squeeze(Ts[0].T)), (Rs[1],np.squeeze(Ts[1].T)), (Rs[2],np.squeeze(Ts[2].T)), (Rs[3],np.squeeze(Ts[3].T))]
# Rts = np.array([(Rs[i],np.squeeze(Ts[i].T)) for i in range(n)])
R,t = util.choose_solution(orb_pts1, orb_pts2, K1, K2, Rts)
P1,P2 = util.camera_matrices(K1, K2, R, t)

uv1 = orb_pts1
uv2 = orb_pts2

# ### Epipolar matching ###
uv1 = np.array([orb_kp1[i].pt for i in range(len(orb_kp1))])
uv2 = util.epipolar_match(img1, img2, F, uv1)

### Triangulate to generate 3D points ###
# X = cv2.triangulatePoints(P1, P2, uv1.T, uv2.T).T
# X /= np.array([X[:,3]]).T

n = len(uv1)
X = np.array([util.linear_triangulation(uv1[i], uv2[i], P1, P2) \
    for i in range(n)])

util.show_point_cloud(X[:,:3],
     xlim=[min(X[:,0])-OFFSET,max(X[:,0])+OFFSET],
     ylim=[min(X[:,1])-OFFSET,max(X[:,1])+OFFSET],
     zlim=[min(X[:,2])-OFFSET,max(X[:,2])+OFFSET])