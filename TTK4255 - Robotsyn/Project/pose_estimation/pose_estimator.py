import sys
sys.path.append("..") # Adds higher directory to python modules path.
from utils.ransac import Ransac
from utils.models import ProjectionDLT
import sim.correspondences as corr
import numpy as np
import numpy.linalg
import cv2

class PoseEstimator:
    def __init__(self, points_img, points_world, intr, dist=np.zeros((4,1))):
        self.points_img = points_img
        self.points_world = points_world
        self.model = ProjectionDLT(points_img, points_world)
        self.intr = intr
        self.dist = dist
        
    def estimate_pose_cv2(self):
        fx, fy, cx, cy = self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy
        camera_matrix = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        # Needs to ensure that the array is stored contiguous in memory
        model_points = np.copy(self.points_world[:,0:3].astype(np.float32), order='C')
        image_points = np.copy(self.points_img[:,0:2].astype(np.float32), order='C')
        dist_coeffs = self.dist
        # (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (success, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs) #, flags=cv2.SOLVEPNP_ITERATIVE)
        theta = np.linalg.norm(rotation_vector)
        k = rotation_vector/theta
        K = PoseEstimator.__skew_symmetric(k)
        # Rodrigues rotation formula
        R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*K**2
        return R, translation_vector
        
    def estimate_pose(self):
        R = np.eye(3)
        t = np.zeros((3, 1))
        return R, t

    @staticmethod
    def __skew_symmetric(v):
        assert len(v) == 3
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]], dtype=np.float32)
