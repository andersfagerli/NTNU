from utilities import *

img = plt.imread('quanser.jpg')

# Camera intrinsics
fx = 1075.47
fy = 1077.22
cx = 621.01
cy = 362.80

cam = CameraIntrinsics(fx, fy, cx, cy)

# Platform dimensions and angles
screw_dist = 0.1145
hinge_dist = 0.325
arm_dist = 0.0552
rotors_dist_x = 0.653
rotors_dist_z = 0.0312

psi = 11.77 * np.pi/180
theta = 28.87 * np.pi/180
phi = -0.5 * np.pi/180

# Transformations
T_platform2cam = np.loadtxt('heli_pose.txt')
T_base2platform = Tx(screw_dist/2) @ Ty(screw_dist/2) @ Rz(psi)
T_hinge2base = Tz(hinge_dist) @ Ry(theta)
T_arm2hinge = Tz(-arm_dist)
T_rotors2arm = Tx(rotors_dist_x) @ Tz(-rotors_dist_z) @ Rx(phi)

# Platform screws
screw1 = point3D(0,0,0)
screw2 = point3D(screw_dist, 0, 0)
screw3 = point3D(0, screw_dist, 0)
screw4 = point3D(screw_dist, screw_dist, 0)

# Fiducial markers
markers = np.loadtxt('heli_points.txt')
marker1 = np.array([[markers[0,0]], [markers[0,1]], [markers[0,2]], [markers[0,3]]])
marker2 = np.array([[markers[1,0]], [markers[1,1]], [markers[1,2]], [markers[1,3]]])
marker3 = np.array([[markers[2,0]], [markers[2,1]], [markers[2,2]], [markers[2,3]]])
marker4 = np.array([[markers[3,0]], [markers[3,1]], [markers[3,2]], [markers[3,3]]])
marker5 = np.array([[markers[4,0]], [markers[4,1]], [markers[4,2]], [markers[4,3]]])
marker6 = np.array([[markers[5,0]], [markers[5,1]], [markers[5,2]], [markers[5,3]]])
marker7 = np.array([[markers[6,0]], [markers[6,1]], [markers[6,2]], [markers[6,3]]])

# Plotting
plt.imshow(img)

drawPoint(T_platform2cam, cam, screw1)
drawPoint(T_platform2cam, cam, screw2)
drawPoint(T_platform2cam, cam, screw3)
drawPoint(T_platform2cam, cam, screw4)

drawPoint(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge, cam, marker1)
drawPoint(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge, cam, marker2)
drawPoint(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge, cam, marker3)
drawPoint(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge @ T_rotors2arm, cam, marker4)
drawPoint(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge @ T_rotors2arm, cam, marker5)
drawPoint(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge @ T_rotors2arm, cam, marker6)
drawPoint(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge @ T_rotors2arm, cam, marker7)

drawFrame(T_platform2cam, cam, scale = 0.1)
drawFrame(T_platform2cam @ T_base2platform, cam, scale = 0.1)
drawFrame(T_platform2cam @ T_base2platform @ T_hinge2base, cam, scale = 0.05)
drawFrame(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge, cam, scale = 0.05)
drawFrame(T_platform2cam @ T_base2platform @ T_hinge2base @ T_arm2hinge @ T_rotors2arm, cam, scale = 0.05)

plt.show()