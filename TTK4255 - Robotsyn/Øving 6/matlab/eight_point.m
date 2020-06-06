function F = eight_point(uv1, uv2)
    % Given n >= 8 point matches, (u1 v1) <-> (u2 v2), compute the
    % fundamental matrix F that satisfies the equations
    %
    %     (u2 v2 1)^T * F * (u1 v1 1) = 0
    %
    % Args:
    %     uv1: (n x 2 array) Pixel coordinates in image 1.
    %     uv2: (n x 2 array) Pixel coordinates in image 2.
    %
    % Returns:
    %     F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1
    %          to lines in image 2.
    %
    % See HZ Ch. 11.2: The normalized 8-point algorithm (p.281).

    % todo: Compute F
    F = zeros([3,3]);
end

function F = closest_fundamental_matrix(F)
    % Computes the closest fundamental matrix in the sense of the
    % Frobenius norm. See HZ, Ch. 11.1.1 (p.280).

    % todo: Compute closest matrix
    F = F;
end
