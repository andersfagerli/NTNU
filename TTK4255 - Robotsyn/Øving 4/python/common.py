import numpy as np

K                  = np.loadtxt('data/cameraK.txt')
p_model            = np.loadtxt('data/model.txt')
platform_to_camera = np.loadtxt('data/pose.txt')

def residuals(uv, weights, yaw, pitch, roll):

    # Helicopter model from Exercise 1 (you don't need to modify this).
    base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
    hinge_to_base    = translate(0, 0, 0.325)@rotate_y(pitch)
    arm_to_hinge     = translate(0, 0, -0.0552)
    rotors_to_arm    = translate(0.653, 0, -0.0312)@rotate_x(roll)
    base_to_camera   = platform_to_camera@base_to_platform
    hinge_to_camera  = base_to_camera@hinge_to_base
    arm_to_camera    = hinge_to_camera@arm_to_hinge
    rotors_to_camera = arm_to_camera@rotors_to_arm

    cx = K[0,2]
    cy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]
    
    X_c_arm = arm_to_camera @ p_model[0:3,:].T
    X_c_rot = rotors_to_camera @ p_model[3:7,:].T
    X_c = np.concatenate((X_c_arm, X_c_rot), axis=1)
    
    u_hat = cx + fx * X_c[0,:] / X_c[2,:]
    v_hat = cy + fy * X_c[1,:] / X_c[2,:]
    uv_hat = np.concatenate((np.array([u_hat]).T,np.array([v_hat]).T), axis=1)
    
    r = (uv_hat - uv)*np.array([weights]).T
    
    return np.linalg.norm(r, axis=1)

def normal_equations(uv, weights, yaw, pitch, roll):
    epsilon = 0.001
    
    r = np.array([residuals(uv, weights, yaw, pitch, roll)]).T
    r_yaw_eps = np.array([residuals(uv, weights, yaw+epsilon, pitch, roll)]).T
    r_pitch_eps = np.array([residuals(uv, weights, yaw, pitch+epsilon, roll)]).T
    r_roll_eps = np.array([residuals(uv, weights, yaw, pitch, roll+epsilon)]).T
    
    r_dyaw = (r_yaw_eps - r) / epsilon
    r_dpitch = (r_pitch_eps - r) / epsilon
    r_droll = (r_roll_eps - r) / epsilon

    J = np.concatenate((r_dyaw,r_dpitch,r_droll), axis=1)
    JTJ = J.T @ J 
    JTr = J.T @ r 
    return JTJ, JTr

def gauss_newton(uv, weights, yaw, pitch, roll):
    #
    # Task 1c: Implement the Gauss-Newton method
    #
    max_iter = 100
    step_size = 0.25
    for iter in range(max_iter):
        JTJ, JTr = normal_equations(uv, weights, yaw, pitch, roll)
        delta = np.linalg.solve(JTJ, -JTr)

        yaw = yaw + step_size * np.squeeze(delta[0])
        pitch = pitch + step_size * np.squeeze(delta[1])
        roll = roll + step_size * np.squeeze(delta[2])
    return yaw, pitch, roll

def levenberg_marquardt(uv, weights, yaw, pitch, roll):
    #
    # Task 2a: Implement the Levenberg-Marquardt method
    #
    xtol = 0.001
    scale = 2
    D = np.identity(3)
    max_iter = 100
    prev_theta = np.array([[yaw], [pitch], [roll]])

    # Initialize lambda / error
    JTJ,_ = normal_equations(uv, weights, yaw, pitch, roll)
    lmda = np.average(JTJ.diagonal()) * 10e-3
    prev_error = np.average(residuals(uv, weights, yaw, pitch, roll))

    for iter in range(max_iter):
        JTJ, JTr = normal_equations(uv, weights, yaw, pitch, roll)
        delta = np.linalg.solve(JTJ + lmda*D, -JTr)

        yaw = yaw + np.squeeze(delta[0])
        pitch = pitch + np.squeeze(delta[1])
        roll = roll + np.squeeze(delta[2])

        new_error = np.max(residuals(uv, weights, yaw, pitch, roll))
        if (new_error <= prev_error):
            lmda = lmda / scale
        else:
            lmda = lmda * scale
        prev_error = new_error

        new_theta = np.array([[yaw], [pitch], [roll]])
        diff = np.abs(new_theta - prev_theta)
        if (np.max(diff) < xtol):
            break
    return yaw, pitch, roll

def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
