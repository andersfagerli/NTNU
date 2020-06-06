function uv2 = epipolar_match(I1, I2, F, uv1)
    % For each point in uv1, finds the matching point in image 2 by
    % an epipolar line search.
    %
    % Args:
    %     I1:  (H x W matrix) Grayscale image 1
    %     I2:  (H x W matrix) Grayscale image 2
    %     F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1 to lines in image 2
    %     uv1: (n x 2 array) Points in image 1
    %
    % Returns:
    %     uv2: (n x 2 array) Best matching points in image 2.
    %
    % Tips:
    % - Image indices must always be integer.
    % - Use round(x) to convert x to an integer.
    % - Use rgb2gray to convert images to grayscale.
    % - Skip points that would result in an invalid access.
    % - Use I(v-w : v+w+1, u-w : u+w) to extract a window of half-width w around (v,u).
    % - Use the sum(..., 'all') function.

    % todo: Compute uv2
    uv2 = zeros(size(uv1));
end
