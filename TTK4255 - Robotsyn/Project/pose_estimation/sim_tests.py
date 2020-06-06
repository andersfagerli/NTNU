import sys
sys.path.append("..") # Adds higher directory to python modules path.
from sim.correspondences import Intrinsics, get_3D_2D_correspondences
from pose_estimation.pose_estimator import PoseEstimator
import cv2

fx, fy, cx, cy = 1000, 1000, 4032 / 2.0, 3024 / 2.0 # Arbitrary
intr = Intrinsics(fx, fy, cx, cy)
uv, body_noise, body_gt, T_true = get_3D_2D_correspondences(intr)
# PoseEstimator:
#     def __init__(self, points_img, points_world, intr, dist=np.zeros((4,1))):
pose_estimator = PoseEstimator(uv, body_noise, intr)
R, t = pose_estimator.estimate_pose_cv2()
t_true = T_true[0:3,-1]
R_true = T_true[0:3, 0:3]
print(f"R: {R}")
print(f"t: {t}")