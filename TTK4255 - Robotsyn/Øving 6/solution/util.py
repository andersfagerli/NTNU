import numpy as np
import matplotlib.pyplot as plt

def draw_line(l, **args):
    """
    Draws the line satisfies the line equation
        x l[0] + y l[1] + l[2] = 0
    clipped to the current plot's box (xlim, ylim).
    """

    def clamp(a, b, a_min, a_max, A, B, C):
        if a < a_min or a > a_max:
            a = np.fmax(a_min, np.fmin(a_max, a))
            b = -(C + a*A)/B
        return a, b

    x_min,x_max = np.sort(plt.xlim())
    y_min,y_max = np.sort(plt.ylim())
    if abs(l[1]) > abs(l[0]):
        x1 = x_min
        x2 = x_max
        y1 = -(l[2] + x1*l[0])/l[1]
        y2 = -(l[2] + x2*l[0])/l[1]
        y1,x1 = clamp(y1, x1, y_min, y_max, l[1], l[0], l[2])
        y2,x2 = clamp(y2, x2, y_min, y_max, l[1], l[0], l[2])
    else:
        y1 = y_min
        y2 = y_max
        x1 = -(l[2] + y1*l[1])/l[0]
        x2 = -(l[2] + y2*l[1])/l[0]
        x1,y1 = clamp(x1, y1, x_min, x_max, l[0], l[1], l[2])
        x2,y2 = clamp(x2, y2, x_min, x_max, l[0], l[1], l[2])
    plt.plot([x1, x2], [y1, y2], **args)

def rgb2gray(I):
    """
    Converts a red-green-blue (RGB) image to grayscale brightness.
    """
    return 0.2989*I[:,:,0] + 0.5870*I[:,:,1] + 0.1140*I[:,:,2]

def show_point_matches(I1, I2, uv1, uv2, F=None):
    """
    Plots k randomly chosen matching point pairs in image 1 and
    image 2. If the fundamental matrix F is given, it also plots the
    epipolar lines.
    """

    k = 10
    sample = np.random.choice(range(len(uv1)), size=k, replace=False)
    uv1 = uv1[sample,:]
    uv2 = uv2[sample,:]

    plt.figure(figsize=(6,4))
    colors = plt.cm.get_cmap('Set1', k).colors
    plt.subplot(121)
    plt.imshow(I1)
    plt.scatter(uv1[:,0], uv1[:,1], s=100, marker='x', c=colors)
    plt.subplot(122)
    plt.imshow(I2)
    plt.scatter(uv2[:,0], uv2[:,1], s=100, marker='o', zorder=10, facecolor='none', edgecolors=colors, linewidths=2)
    if not F is None:
        for i,(u1,v1) in enumerate(uv1):
            l = F@np.array((u1,v1,1))
            draw_line(l, linewidth='1', color=colors[i])
    plt.tight_layout()

def show_point_cloud(X, xlim, ylim, zlim):
    """
    Creates a mouse-controllable 3D plot of the input points.
    """
    plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')

    # This could be changed to use scatter if you want to
    # provide a per-point color. Otherwise, the plot function
    # is much faster.
    ax.plot(X[:,0], X[:,2], X[:,1], '.')

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('x')
    ax.set_zlabel('y')
    ax.set_ylabel('z')
    plt.show()
