import sys
sys.path.append("../pointcloud")

# External libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import collections

# Own libraries
import SIFT as sift
from read_write_model import read_model, qvec2rotmat
from export_to_colmap import export_to_colmap
from read_descriptors import readDescriptors
import util

"""
NB! MUST BE DONE BEFOREHAND

Use export_to_colmap.py to do feature extraction on a set of images and export
the generated .txt files to COLMAP. In COLMAP, do feature matching and reconstruction,
and export the model as binary files. Place the binary files in the same folder
the data (images + .txt files).
"""

### Parameters ###
data_path = 'data/'
image_naming = 'out'
image_format = 'png'
num_images = 30
nfeatures = 0

K = np.loadtxt('camera/K_Anders.txt')

### Reading data from COLMAP binary files ###
cameras, images, points3D = read_model(data_path, ".bin")

### Retrieve descriptors ###
descriptors = readDescriptors(data_path, image_naming, image_format, num_images)

### Generate pointcloud with descriptors ###
PointCloud = collections.namedtuple("PointCloud", ["xyz", "rgb", "des"])

xyz = np.empty((len(points3D), 3))
des = np.empty((xyz.shape[0], 128))
rgb = np.zeros((len(points3D), 3))

for (i, point_id) in enumerate(points3D):
    image_id = points3D[point_id].image_ids[0]
    feature_id = np.squeeze(np.where(images[image_id].point3D_ids == point_id))
    if feature_id.size > 1:
        feature_id = feature_id[0]
    
    xyz[i,:] = points3D[point_id].xyz
    rgb[i,:] = (points3D[point_id].rgb[0], points3D[point_id].rgb[1], points3D[point_id].rgb[2])
    des[i,:] = descriptors[image_id][feature_id]

pointcloud = PointCloud(xyz=xyz, rgb=rgb, des=np.float32(des))

### Query image ###
query_img = cv2.imread('data/query.png', 0)
mask = np.zeros(query_img.shape[:2], dtype=np.uint8)
start_y = int(query_img.shape[0]/10)
start_x = int(query_img.shape[1]/4)
cv2.rectangle(mask, (start_x, start_y), (query_img.shape[1] - start_x, query_img.shape[0] - start_y), (255), thickness=-1)

kp, des = sift.keypointAndDescriptor(query_img, nfeatures=nfeatures)

pts_img, pts3D, _, _ = sift.flannMatches3D2D(kp, des, pointcloud.xyz, pointcloud.des)

success, rot_vec, translation, inliers = cv2.solvePnPRansac(pts3D, pts_img, K, None)
# success, rot_vec, translation = cv2.solvePnP(pts3D, pts_img, K, None)

# Shuffle terms so they coincide with our coordinate system
rot_vec = np.array([rot_vec[0], rot_vec[2], rot_vec[1]])
R,_ = cv2.Rodrigues(rot_vec)

translation = np.array([-translation[0], translation[2], -translation[1]])

T = util.getTransformation(R, translation)

### Plotting ###

# Show pose estimation
OFFSET = 20
xlim = [-OFFSET/2,+OFFSET/2]
ylim = [-OFFSET/2,+OFFSET/2]
zlim = [20,40]

plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

ax.scatter(xyz[:,0], xyz[:,2], xyz[:,1], s=1, c=rgb/255.0)

util.draw_origin(ax, scale=15)
util.draw_frame(ax, T, scale=5)

for im_id in images:
    # Get rotation
    R = qvec2rotmat(images[im_id].qvec)
    rot_vec,_ = cv2.Rodrigues(R)
    rot_vec = np.array([rot_vec[0], rot_vec[2], rot_vec[1]])
    R,_ = cv2.Rodrigues(rot_vec)
    # Get translation
    tvec = images[im_id].tvec
    t = np.array([[-tvec[0]], [tvec[2]], [-tvec[1]]])
    
    T = util.getTransformation(R, t)
    util.draw_frame(ax, T, scale=5)

ax.set_xlim(xlim)
ax.set_ylim(zlim)
ax.set_zlim([ylim[1], ylim[0]])
ax.set_xlabel('x')
ax.set_zlabel('y')
ax.set_ylabel('z')
plt.show()

# # Show 2D-3D inlier matches
# sample = np.random.choice(range(len(pts3D)), size=10, replace=False)

# plt.figure(figsize=(10,10))
# ax = plt.axes(projection='3d')
# ax.plot(xyz[:,0], xyz[:,2], xyz[:,1], '.', markersize=1)
# ax.plot(pts3D[sample,0], pts3D[sample,2], pts3D[sample,1], '.', markersize=2, color='red')
# ax.set_xlim(xlim)
# ax.set_ylim(zlim)
# ax.set_zlim([ylim[1], ylim[0]])
# ax.set_xlabel('x')
# ax.set_zlabel('y')
# ax.set_ylabel('z')

# plt.show()

# plt.figure(figsize=(10,10))
# im = plt.imread('data/query.png')
# plt.imshow(im)
# plt.plot(pts_img[sample,0], pts_img[sample,1], '.', markersize=5, color='red')
# plt.show()
