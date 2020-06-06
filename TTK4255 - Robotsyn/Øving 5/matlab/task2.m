clear;
edge_threshold = 0; % todo: choose an appropriate value
blur_sigma     = 0; % todo: choose an appropriate value
line_threshold = 10; % todo: choose an appropriate value
bins           = 100; % todo: choose an appropriate value
filename       = '../data/image1_und.jpg';

I_rgb       = imread(filename);
I_rgb       = im2double(I_rgb);
I_gray      = rgb2gray(I_rgb);
I_blur      = blur(I_gray, blur_sigma);
[Iu,Iv,Im]  = central_difference(I_blur);
[u,v,theta] = extract_edges(Iu, Iv, Im, edge_threshold);

%
% Task 2a: Compute accumulator array H
%
rho_max   = 0; % Placeholder
rho_min   = 0; % Placeholder
theta_min = 0; % Placeholder
theta_max = 0; % Placeholder
H = zeros(bins,bins); % Placeholder

% Tip: Use histcounts2 for task 2a:
% [H, ~, ~] = histcounts2(theta, rho, bins, ...
%     'XBinLimits', [theta_min, theta_max], ...
%     'YBinLimits', [rho_min, rho_max]);
% H = H'; % Make rows be rho and columns be theta

%
% Task 2b: Find local maxima
%
window_size = 11;
[peak_rows,peak_cols] = extract_peaks(H, window_size, line_threshold);

%
% Task 2c: Convert peak (row, column) pairs into (theta, rho) pairs.
%
peak_theta = [0 0.2 0.5 0.7]'; % Placeholder to demonstrate use of draw_line
peak_rho   = [10 100 300 500]'; % Placeholder to demonstrate use of draw_line

subplot(211);
imagesc(H, 'XData', [theta_min theta_max], 'YData', [rho_min rho_max]);
xlabel('\theta (radians)');
ylabel('\rho (pixels)');
cb = colorbar();
cb.Label.String = 'Votes';
subplot(212);
imshow(I_rgb); hold on;
for i=1:size(peak_rho)
    draw_line(peak_theta(i), peak_rho(i));
end
