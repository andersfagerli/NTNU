function [P1,P2] = camera_matrices(K1, K2, R, t)
    % Computes the projection matrix for camera 1 and camera 2.
    %
    % Args:
    %     K1,K2: (3x3 matrix) Intrinsic matrix for camera 1 and camera 2.
    %     R,t: The rotation and translation mapping points in camera 1 to points in camera 2.
    %
    % Returns:
    %     P1,P2: The projection matrices with shape 3x4.

    % todo: Compute the correct matrices
    P1 = zeros([3,4]);
    P2 = zeros([3,4]);
end
