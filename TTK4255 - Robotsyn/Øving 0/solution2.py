import matplotlib.pyplot as plt
import numpy as np

def point(x,y):
    return np.array([[x], [y], [1]])

def rotate(degrees):
    c = np.cos(degrees*np.pi/180)
    s = np.sin(degrees*np.pi/180)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def translate(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])

##################################
# Task c)
##################################
p1_a = point(1,0)
p2_b = point(1,0)
p3_b = point(0.5,0.5)
b_to_a = rotate(45)
a_to_b = b_to_a.T # inverse of rotation matrix is its transpose
p1_b = a_to_b @ p1_a
p2_a = b_to_a @ p2_b
p3_a = b_to_a @ p3_b
print(p1_b) # 0.707, -0.707
print(p2_a) # 0.707,  0.707
print(p3_a) # 0.000,  0.707

##################################
# Task e,f,g)
##################################
def draw_line(a, b, **args):
    plt.plot([a[0,0], b[0,0]], [a[1,0], b[1,0]], **args)

def draw_frame(T, label):
    origin = T@point(0,0)
    draw_line(origin, T@point(1,0), color='red')
    draw_line(origin, T@point(0,1), color='green')
    plt.text(origin[0,0], origin[1,0] - 0.4, label)

plt.figure(figsize=(8,3))

# Problem 1
T = rotate(30)@translate(3,0)
plt.subplot(131)
draw_frame(T, 'a')
plt.axis('scaled')
plt.grid()
plt.xlim([-1,5])
plt.ylim([-1,5])
plt.xticks([-1,0,1,2,3,4,5])
plt.yticks([-1,0,1,2,3,4,5])

# Problem 2
T = translate(2,1)@rotate(45)
plt.subplot(132)
draw_frame(T, 'a')
plt.axis('scaled')
plt.grid()
plt.xlim([-1,5])
plt.ylim([-1,5])
plt.xticks([-1,0,1,2,3,4,5])
plt.yticks([-1,0,1,2,3,4,5])

# Problem 3
T_a = rotate(30)@translate(1.5,0)
T_b = translate(0,3)@T_a@rotate(15)
T_c = rotate(-45)@T_b

plt.subplot(133)
draw_frame(T_a, 'a')
draw_frame(T_b, 'b')
draw_frame(T_c, 'c')
plt.axis('scaled')
plt.grid()
plt.xlim([-1,5])
plt.ylim([-1,5])
plt.xticks([-1,0,1,2,3,4,5])
plt.yticks([-1,0,1,2,3,4,5])

plt.tight_layout()
plt.show()
