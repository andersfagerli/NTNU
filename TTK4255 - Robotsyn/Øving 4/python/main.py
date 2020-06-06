import numpy as np
import matplotlib.pyplot as plt
from common import *

all_markers = np.loadtxt('data/markers.txt')
m = 7 # Number of markers (not all are detected in each frame)

# Task 1
yaw   = 11.6*np.pi/180
pitch = 28.9*np.pi/180
roll  = -0.6*np.pi/180

# Task 2
# yaw = 0
# pitch = 0
# roll = 0

# Task 1
# method = gauss_newton
# last_image = 86

# Task 2
method = levenberg_marquardt
last_image = 360

trajectory = []
for image_number in range(last_image + 1):
    markers = all_markers[image_number,:]
    markers = np.reshape(markers, [m, 3])
    weights = markers[:,0] # weight = 1 if marker was detected or 0 otherwise
    uv = markers[:, 1:3]
    if image_number == 0:
        r = residuals(uv, weights, yaw, pitch, roll)
        print('Residuals on image are:', r)
    yaw, pitch, roll = method(uv, weights, yaw, pitch, roll)
    trajectory.append([yaw, pitch, roll])
trajectory = np.array(trajectory)

#
# Generate output plot comparing encoder values against vision estimate
#
logs       = np.loadtxt('data/logs.txt')
log_time   = logs[:,0]
log_yaw    = logs[:,1]
log_pitch  = logs[:,2]
log_roll   = logs[:,3]
video_fps  = 16
video_time = np.arange(last_image + 1)/video_fps
plt.figure(figsize=[6,6])
plt.subplot(311)
plt.plot(log_time, log_yaw, color='#999999', linewidth=8, label='Encoder log')
plt.plot(video_time, trajectory[:,0], color='black', label='Vision estimate')
plt.legend()
plt.xlim([0, last_image/video_fps])
plt.ylim([-1, 1])
plt.ylabel('Yaw')
plt.subplot(312)
plt.plot(log_time, log_pitch, color='#999999', linewidth=8)
plt.plot(video_time, trajectory[:,1], color='black')
plt.xlim([0, last_image/video_fps])
plt.ylim([-0.25, 0.6])
plt.ylabel('Pitch')
plt.subplot(313)
plt.plot(log_time, log_roll, color='#999999', linewidth=8)
plt.plot(video_time, trajectory[:,2], color='black')
plt.xlim([0, last_image/video_fps])
plt.ylim([-0.6, 0.6])
plt.ylabel('Roll')
plt.xlabel('Time (Seconds)')
plt.tight_layout()
plt.savefig('out.png')
plt.show()
