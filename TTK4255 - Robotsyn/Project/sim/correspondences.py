import sys
sys.path.append("..") # Adds higher directory to python modules path.
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from sim import body
from sim.utils import *
import types

class Intrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @classmethod
    def from_file(cls, source_file):     
        with open(source_file) as fh:
            for line in fh:
                words = line.split()
                if not words:
                    break
                if words[0] == "fx":
                    fx = float(words[2])
                elif words[0] == "fy":
                    fy = float(words[2])
                elif words[0] == "cx":
                    cx = float(words[2])
                elif words[0] == "cy":
                    cy = float(words[2])
        return cls(fx, fy, cx, cy)
    
    @property
    def K(self):
        K = np.array([[self.fx,       0, self.cx],
                      [      0, self.fy, self.cy],
                      [      0,       0,       1]])
        return K

    def __str__(self):
        return "fx: {}\nfy: {}\ncx: {}\ncy: {}".format(self.fx, self.fy, self.cx, self.cy)

def get_camera_pose(body_pose):
    T = translate(5, 0, 0)
    return (T @ body_pose.T).T

def get_camera_center(camera_pose):
    center = np.average(camera_pose[:,0:3], axis=0)
    return center

def project(body_pose, intr):
    P = intr.K @ np.block([np.eye(3), np.zeros((3, 1))])
    uv = P @ body_pose.T
    uv /= uv[-1,:]
    # c = uv.shape[1]
    # uv = uv[:,1:c]
    return uv.T

def invert(T):
    R = T[0:3,0:3]
    t = np.array([T[0:3,-1]]).T
    inv_T = np.block([[R.T, -R.T @ t],
                      [0, 0, 0, 1]])
    return inv_T

def gen_outliers(center, mu_r=1.25, std_r=0.75, num_outliers=5):
    r = np.array([np.random.normal(mu_r, std_r, num_outliers)]).T
    angles = np.array([np.random.uniform(0, 2*np.pi, num_outliers)]).T
    outliers = np.block([np.zeros((num_outliers, 1)), r*np.cos(angles), r*np.sin(angles), np.ones((num_outliers, 1))])
    outliers += center
    return outliers
    
def get_3D_2D_correspondences(intr):
    # Get 3D body ground truth and noisy
    body_trans, T = body.gen_3d_body_transformed(x=3) #(translate(offset, 0, 0) @ body.
    center = np.average(body_trans[0:3], axis=0)
    body_prtb = body.gen_noisy_3d_points(body_trans)
    body_prtb /= body_prtb[:,-1].reshape((-1, 1))

    # Add outliers and shuffle
    outliers = gen_outliers(center)
    body_prtb = np.block([[body_prtb],
                          [outliers]])
    np.random.shuffle(body_prtb)
    body_prtb /= body_prtb[:,-1].reshape((-1, 1))
    
    # Needs to transform between camera and world frame
    T_wc = (rotate_x(-np.pi / 2.0) @ rotate_y(np.pi / 2.0)).T
    T_wc[np.abs(T_wc) < 1e-5] = 0
    body_rel_camera = (T_wc @ body_prtb.T).T

    # Generate 2D correspondences
    uv = np.round(project(body_rel_camera, intr)).astype(np.int32)
    
    return uv, body_prtb, body_trans, T

def plot_coordinate_frame_3D(O, ax):
    x = O[0:2, 1] - O[0:2, 0]
    y = O[0:2, 3] - O[0:2, 2]
    z = O[0:2, 5] - O[0:2, 4]

    scaling = 1.1
    tx = O[0:2, 1] * scaling 
    ty = O[0:2, 3] * scaling
    tz = O[0:2, 5] * scaling

    ax.quiver(intr.cx, intr.cy, x[0,], x[1,], color='r', angles='xy', scale_units='xy', scale=1.)
    ax.quiver(intr.cx, intr.cy, y[0,], y[1,], color='g', angles='xy', scale_units='xy', scale=1.)
    ax.quiver(intr.cx, intr.cy, z[0,], z[1,], color='b', angles='xy', scale_units='xy', scale=1.)
    ax.text(tx[0], tx[1], 'x', color='r', fontsize=12)
    ax.text(ty[0], ty[1], 'y', color='g', fontsize=12)
    ax.text(tz[0], tz[1], 'z', color='b', fontsize=12)

if __name__ == "__main__":
    fx, fy, cx, cy = 1000, 1000, 4032 / 2.0, 3024 / 2.0 # Arbitrary
    intr = Intrinsics(fx, fy, cx, cy)
    uv, body_noise, body_gt = get_3D_2D_correspondences(intr)
    plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.plot(body_noise[:,0], body_noise[:,1], body_noise[:,2], '.')
    ax.plot(body_gt[:,0], body_gt[:,1], body_gt[:,2])
    ax.quiver(0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1], length=1, color='rgb')
    plt.figure()
    plt.plot(uv[:,0], uv[:,1], '.')
    plt.plot([0, 4032, 4032, 0, 0], [0, 0, 3024, 3024, 0])
    plt.ylim([3124, -100])
    plt.show()
    