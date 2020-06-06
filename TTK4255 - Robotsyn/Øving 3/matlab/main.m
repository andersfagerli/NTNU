clear;
K           = load('../data/cameraK.txt');
all_markers = load('../data/markers.txt');
XY          = load('../data/model.txt');
n           = size(XY,1);
for image_number=0:22
    I = imread(sprintf('../data/video%04d.jpg', image_number));
    markers = all_markers(image_number + 1,:)';
    markers = reshape(markers, [3, n])';
    matched = markers(:,1) == 1;
    uv = markers(matched, 2:3);
    
    % Convert pixel coordinates to normalized image coordinates
    xy = (uv - [K(1,3) K(2,3)]) ./ [K(1,1) K(2,2)];

    H = estimate_H(xy, XY(matched, 1:2));
    [T1,T2] = decompose_H(H);
    T = choose_solution(T1, T2);

    % Compute predicted corner locations using model and homography
    uv_pred = (K*H*XY')';
    uv_pred = uv_pred ./ uv_pred(:,3);

    clf();
    imshow(I, 'Interpolation', 'bilinear'); hold on;
    scatter(uv(:,1), uv(:,2), 'yellow', 'filled');
    scatter(uv_pred(:,1), uv_pred(:,2), 'red');
    draw_frame(K, T, 7);
    legend('Observed', 'Predicted');
    print(sprintf('../data/out%04d.png', image_number), '-dpng');
end

function H = estimate_H(xy, XY)
    %
    % Task 2
    %
    H = eye(3); % Placeholder code
end

function [T1,T2] = decompose_H(H)
    %
    % Task 3a
    %
    T1 = eye(4); % Placeholder code
    T2 = eye(4); % Placeholder code
end

function T = choose_solution(T1, T2)
    %
    % Task 3b
    %
    T = T1; % Placeholder code
end
