import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sim.utils import *


def perturb_trans(mu=0, std_t=0.01, std_R=0.5):
    dt = np.random.normal(mu, std_t, 3)
    dR = np.random.normal(mu, std_R * np.pi / 180.0, 3)
    dT = (rotate_z(dR[2]) @ translate(0, 0, dt[2])) @ (rotate_y(dR[1]) @ translate(0, dt[1], 0)) @ (rotate_x(dR[0]) @ translate(dt[0], 0, 0))
    return dT
    

def gen_3d_body_untransformed(size=12):
    """ Generate a simple circle about the x-axis """
    angles = np.array([np.linspace(0, 2*np.pi, num=size)]).T
    points = np.block([np.zeros((size, 1)), np.cos(angles), np.sin(angles), np.ones((size, 1))])
    return points

def gen_body_transform(x=1, y=1, z=1, angle=10):
     t = translate(x, y, z)
     angle = angle * np.pi / 180.0
     R = rotate_z(angle) @ rotate_y(angle) @ rotate_x(angle)
     T = t @ R
     return T

def gen_noisy_3d_points(body_trans):
    body_prtb = np.array([perturb_trans() @ point for point in body_trans])
    return body_prtb

def gen_3d_body_transformed(x=1, y=1, z=1, angle=10):
    T = gen_body_transform(x=x, y=y, z=z, angle=angle)
    body_untrans = gen_3d_body_untransformed()
    body_trans = (T @ body_untrans.T).T
    return body_trans, T
    
if __name__ == "__main__":
    T = gen_body_transform()
    body_untrans = gen_3d_body_untransformed()
    body_trans = (T @ body_untrans.T).T
    body_prtb = np.array([perturb_trans() @ point for point in body_trans])
    ax = plt.axes(projection='3d')
    ax.plot(body_untrans[:,0], body_untrans[:,1], body_untrans[:,2])
    ax.plot(body_trans[:,0], body_trans[:,1], body_trans[:,2])
    ax.plot(body_prtb[:,0], body_prtb[:,1], body_prtb[:,2])
    plt.show()