from utilities import *

box_raw = np.loadtxt('box.txt')

# Camera intrinsics
fx = 1000
fy = 1100
cx = 320
cy = 240

cam = CameraIntrinsics(fx, fy, cx, cy)

# World parameters
Z = 5
theta = 30 * np.pi/180

# Box dimensions
box_x = 2
box_y = 1
box_z = 2

# Translation/Rotation
T_a = Tz(Z+box_z/2)
T_b = Tz(Z+box_z/2) @ Rx(theta) @ Ry(theta)

box_a = T_a @ box_raw.T
box_b = T_b @ box_raw.T

# Image formation
u_a = cx + fx * box_a[0,] / box_a[2,]
v_a = cy + fy * box_a[1,] / box_a[2,]

u_b = cx + fx * box_b[0,] / box_b[2,]
v_b = cy + fy * box_b[1,] / box_b[2,]

# Plotting
plt.figure(figsize=(4,6))

plt.subplot(211)
plt.ylim([480, 0])
plt.scatter(u_a, v_a, marker='.', c='black')

plt.subplot(212)
plt.ylim([480, 0])
plt.scatter(u_b, v_b, marker='.', c='black')

drawFrame(T_b, cam)

plt.show()