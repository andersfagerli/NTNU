import matplotlib.pyplot as plt
import numpy as np

class CameraIntrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

def Tx(tx):
    matrix = np.array([
        [1,0,0,tx],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return matrix

def Ty(ty):
    matrix = np.array([
        [1,0,0,0],
        [0,1,0,ty],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return matrix

def Tz(tz):
    matrix = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,tz],
        [0,0,0,1]
    ])
    return matrix

def Rx(theta):
    matrix = np.array([
        [1,0,0,0],
        [0,np.cos(theta), -np.sin(theta),0],
        [0,np.sin(theta), np.cos(theta), 0],
        [0,0,0,1]
    ])
    return matrix

def Ry(theta):
    matrix = np.array([
        [np.cos(theta),0,np.sin(theta),0],
        [0,1,0,0],
        [-np.sin(theta),0, np.cos(theta), 0],
        [0,0,0,1]
    ])
    return matrix

def Rz(theta):
    matrix = np.array([
        [np.cos(theta),-np.sin(theta),0,0],
        [np.sin(theta),np.cos(theta),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return matrix

def point2D(x,y):
    return np.array([[x], [y], [1]])

def point3D(x,y,z):
    return np.array([[x], [y], [z], [1]])

def imageFormation(point,fx,fy,cx,cy):
    u = cx + fx * point[0,] / point[2,]
    v = cy + fy * point[1,] / point[2,]

    return point2D(u,v)

def drawPoint(T, cam, point_3D, **args):
    point_2D = imageFormation(T @ point_3D, cam.fx, cam.fy, cam.cx, cam.cy)
    plt.scatter(point_2D[0], point_2D[1], marker = '.', c = 'white')

def drawLine(T, cam, pointA_3D, pointB_3D, **args):
    x = T @ pointA_3D
    y = T @ pointB_3D

    pointA_2D = imageFormation(x, cam.fx, cam.fy, cam.cx, cam.cy)
    pointB_2D = imageFormation(y, cam.fx, cam.fy, cam.cx, cam.cy)

    plt.plot([pointA_2D[0,0], pointB_2D[0,0]], [pointA_2D[1,0], pointB_2D[1,0]], **args)

def drawFrame(T, cam, scale = 1.0):
    ori = point3D(0,0,0)
    x = point3D(1*scale,0,0)
    y = point3D(0,1*scale,0)
    z = point3D(0,0,1*scale)

    drawLine(T, cam, ori, x)
    drawLine(T, cam, ori, y)
    drawLine(T, cam, ori, z)
