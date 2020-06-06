function X = linear_triangulation(uv1, uv2, P1, P2)
    % Compute the 3D position of a point from 2D correspondences.
    %
    % Args:
    %    uv1:    2D projection of the point in image 1
    %    uv2:    2D projection of the point in image 2
    %    P1:     Projection matrix with shape 3 x 4 for image 1
    %    P2:     Projection matrix with shape 3 x 4 for image 2
    %
    % Returns:
    %    X:      3D coordinates of point in the camera frame of image 1
    %            (not homogeneous!)
    %
    % See HZ Ch. 12.2: Linear triangulation methods (p312)

    % todo: Compute X
    X = [0 0 0]';
end
