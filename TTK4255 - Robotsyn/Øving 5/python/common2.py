import matplotlib.pyplot as plt
import numpy as np

def extract_peaks(arr, window_size, threshold):
    h = window_size//2
    dilated = np.zeros_like(arr)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            col0 = max(0, col - h)
            col1 = min(arr.shape[1], col + h + 1)
            row0 = max(0, row - h)
            row1 = min(arr.shape[0], row + h + 1)
            window = arr[row0:row1, col0:col1]
            dilated[row,col] = np.amax(window)

    maxima = np.logical_and(dilated == arr, arr >= threshold)
    peak_rows,peak_cols = np.nonzero(maxima)
    return peak_rows,peak_cols

def draw_line(theta, rho, **args):
    """
    Draws a line given in normal form (rho, theta).
    Uses the current plot's xlim and ylim as bounds.
    """

    def clamp(a, b, a_min, a_max, rho, A, B):
        if a < a_min or a > a_max:
            a = np.fmax(a_min, np.fmin(a_max, a))
            b = (rho-a*A)/B
        return a, b

    x_min,x_max = np.sort(plt.xlim())
    y_min,y_max = np.sort(plt.ylim())
    c = np.cos(theta)
    s = np.sin(theta)
    if np.fabs(s) > np.fabs(c):
        x1 = x_min
        x2 = x_max
        y1 = (rho-x1*c)/s
        y2 = (rho-x2*c)/s
        y1,x1 = clamp(y1, x1, y_min, y_max, rho, s, c)
        y2,x2 = clamp(y2, x2, y_min, y_max, rho, s, c)
    else:
        y1 = y_min
        y2 = y_max
        x1 = (rho-y1*s)/c
        x2 = (rho-y2*s)/c
        x1,y1 = clamp(x1, y1, x_min, x_max, rho, c, s)
        x2,y2 = clamp(x2, y2, x_min, x_max, rho, c, s)
    plt.plot([x1, x2], [y1, y2], **args)
