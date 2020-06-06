#                              README
# This script expects you to have filled out the functions declared
# in common1.py: blur, central_difference and extract_edges. If you
# define these according to the expected input and output, you should
# be able to simply run this file to generate the figure for task 1.
#
import numpy as np
import matplotlib.pyplot as plt
from common1 import *

edge_threshold = 0.1
blur_sigma     = 1
filename       = '../data/image1_und.jpg'

I_rgb      = plt.imread(filename)
I_rgb      = I_rgb/255.0
I_gray     = rgb2gray(I_rgb)
I_blur     = blur(I_gray, blur_sigma)
Iu, Iv, Im = central_difference(I_blur)
u,v,theta  = extract_edges(Iu, Iv, Im, edge_threshold)

fig, axes = plt.subplots(1,5,figsize=[15,4], sharey='row', sharex='row')
plt.set_cmap('gray')
axes[0].imshow(I_blur)
axes[1].imshow(Iu, vmin=-0.05, vmax=0.05)
axes[2].imshow(Iv, vmin=-0.05, vmax=0.05)
axes[3].imshow(Im, vmin=+0.00, vmax=0.10, interpolation='bilinear')
edges = axes[4].scatter(u, v, s=1, c=theta, cmap='hsv')
fig.colorbar(edges, ax=axes[4], orientation='horizontal', label='$\\theta$ (radians)')
for a in axes:
    a.set_xlim([300, 600])
    a.set_ylim([I_rgb.shape[0], 0])
    a.set_aspect('equal')
axes[0].set_title('Blurred input')
axes[1].set_title('Gradient in u')
axes[2].set_title('Gradient in v')
axes[3].set_title('Gradient magnitude')
axes[4].set_title('Extracted edges')
plt.tight_layout()
plt.savefig('out1.png')
plt.show()
