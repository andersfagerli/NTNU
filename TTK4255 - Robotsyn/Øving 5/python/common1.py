import numpy as np

# Task 1a
def central_difference(I):
    """
    Computes the gradient in the u and v direction using
    a central difference filter, and returns the resulting
    gradient images (Iu, Iv) and the gradient magnitude Im.
    """
    Iu = np.zeros_like(I) # Placeholder
    Iv = np.zeros_like(I) # Placeholder
    Im = np.zeros_like(I) # Placeholder
    rows,cols = I.shape
    kernel = np.array([0.5, 0, -0.5])

    for i in range(rows):
        Iu[i,:] = np.convolve(I[i,:], kernel, mode='same')
    for j in range(cols):
        Iv[:,j] = np.convolve(I[:,j], kernel, mode='same')
    Im = np.sqrt(Iu**2 + Iv**2)
    
    return Iu, Iv, Im

# Task 1b
def blur(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """
    w = 2*np.ceil(3*sigma) + 1
    kernel = np.linspace(-(w-1)/2, (w-1)/2, w)
    rows,cols = I.shape
    result = np.zeros_like(I)
    for i in range(rows):
        result[i,:] = np.convolve(I[i,:], kernel, mode='same')
    for j in range(cols):
        result[:,j] = np.convolve(I[:,j], kernel, mode='same')
    
    return result

# Task 1c
def extract_edges(Iu, Iv, Im, threshold):
    """
    Returns the u and v coordinates of pixels whose gradient
    magnitude is greater than the threshold.
    """

    # This is an acceptable solution for the task (you don't
    # need to do anything here). However, it results in thick
    # edges. If you want better results you can try to replace
    # this with a thinning algorithm as described in the text.
    v,u = np.nonzero(Im > threshold)
    theta = np.arctan2(Iv[v,u], Iu[v,u])
    return u, v, theta

def rgb2gray(I):
    """
    Converts a red-green-blue (RGB) image to grayscale brightness.
    """
    return 0.2989*I[:,:,0] + 0.5870*I[:,:,1] + 0.1140*I[:,:,2]
