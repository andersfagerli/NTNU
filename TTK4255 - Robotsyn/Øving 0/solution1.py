import matplotlib.pyplot as plt
import numpy as np

rgb = plt.imread('roomba.jpg')
h,w = rgb.shape[0:2]
print('width: %d\nheight: %d' % (w,h))
rgb = rgb/255.0 # Convert to floating-point to prevent overflow

# Compute distance in RGB space to strong red
difference = rgb - np.array([1.0,0.0,0.0])
distance = np.linalg.norm(difference, axis=2) # Euclidean length (L2 norm)of the third dimension (rgb difference)
thresholded = distance < 0.7 # Isolate pixels that are sufficiently close to strong red

plt.figure(figsize=(6,4))
plt.subplot(221)
plt.imshow(rgb)
plt.title('Input RGB')
plt.subplot(222)
plt.imshow(rgb[:,:,0], cmap='gray')
plt.title('R channel')
plt.subplot(223)
plt.imshow(rgb[:,:,0] > 0.6, cmap='gray')
plt.title('Pixels with R > 0.6')
plt.subplot(224)
plt.imshow(thresholded, cmap='gray')
plt.title('Pixels with ||rgb - [1 0 0]|| < 0.7')
plt.tight_layout()
plt.show()
