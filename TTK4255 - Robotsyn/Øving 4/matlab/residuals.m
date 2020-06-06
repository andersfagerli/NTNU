function r = residuals(K, platform_to_camera, p_model, uv, weights, yaw, pitch, roll)
    base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)*rotate_z(yaw);
    hinge_to_base    = translate(0, 0, 0.325)*rotate_y(pitch);
    arm_to_hinge     = translate(0, 0, -0.0552);
    rotors_to_arm    = translate(0.653, 0, -0.0312)*rotate_x(roll);
    base_to_camera   = platform_to_camera*base_to_platform;
    hinge_to_camera  = base_to_camera*hinge_to_base;
    arm_to_camera    = hinge_to_camera*arm_to_hinge;
    rotors_to_camera = arm_to_camera*rotors_to_arm;

    %
    % Task 1a: Implement the rest of this function
    %

    % Tip: If A is an Nx2 matrix, vecnorm(A, 2, 2)
    % computes the Euclidean length of each row and
    % returns an Nx1 matrix.

    m = size(p_model,1); % Placeholder
    r = inf*ones(m, 1);  % Placeholder
end
