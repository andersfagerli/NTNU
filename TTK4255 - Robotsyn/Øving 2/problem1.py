import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# Camera parameters/distortion coefficients
fx = 9.842439e+02
cx = 6.900000e+02
fy = 9.808141e+02
cy = 2.331966e+02
k1 = -3.728755e-01
k2 = 2.037299e-01
k3 = -7.233722e-02
p1 = 2.219027e-03
p2 = 1.383707e-03

dist_img = plt.imread('data/kitti.jpg')
undist_img = np.array(dist_img)

height, width, depth = dist_img.shape

x = np.array([(np.arange(0, width) - cx) / fx])
y = np.array([(np.arange(0, height) - cy) / fy]).T

X = np.matlib.repmat(x, height, 1)
Y = np.matlib.repmat(y, 1, width)

R = np.sqrt(X**2 + Y**2)

delta_x = (k1*R**2 + k2*R**4 + k3*R**6)*X + 2*p1*X*Y + p2*(R**2 + 2*X**2)
delta_y = (k1*R**2 + k2*R**4 + k3*R**6)*Y + p1*(R**2 + 2*Y**2) + 2*p2*X*Y

u_src = cx + fx*(X + delta_x)
v_src = cy + fy*(Y + delta_y)

for i in range(np.size(dist_img,0)):
    for j in range(np.size(dist_img,1)):
        undist_img[i,j,:] = dist_img[int(v_src[i,j]), int(u_src[i,j]),:]

plt.imshow(undist_img)
plt.show()