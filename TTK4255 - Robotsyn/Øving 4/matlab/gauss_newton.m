function [yaw, pitch, roll] = gauss_newton(K, platform_to_camera, p_model, uv, weights, yaw, pitch, roll)
    %
    % Task 1c: Implement the Gauss-Newton method
    %
    % This will involve calling the functions you defined in a and b.
    
    max_iter = 100;
    step_size = 0.25;
    for iter=1:max_iter
        yaw   = yaw;   % Placeholder
        pitch = pitch; % Placeholder
        roll  = roll;  % Placeholder
    end
end