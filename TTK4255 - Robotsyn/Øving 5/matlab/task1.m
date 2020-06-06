%                              README
% This script expects you to have filled out the following functions:
% blur, central_difference and extract_edges. If you define these 
% according to the expected input and output, you should be able to 
% simply run this file to generate the figure for task 1.
%
clear;
filename       = '../data/image1_und.jpg';
edge_threshold = 0; % todo: choose an appropriate value
blur_sigma     = 0; % todo: choose an appropriate value

I_rgb       = imread(filename);
I_rgb       = im2double(I_rgb);
I_gray      = rgb2gray(I_rgb);
I_blur      = blur(I_gray, blur_sigma);
[Iu,Iv,Im]  = central_difference(I_blur);
[u,v,theta] = extract_edges(Iu, Iv, Im, edge_threshold);

figure(1);
set(gcf,'Position',[100 100 1000 300])
subplot(151); imshow(I_blur);            xlim([300, 500]); title('Blurred input');
subplot(152); imshow(Iu, [-0.05, 0.05]); xlim([300, 500]); title('Gradient in u');
subplot(153); imshow(Iv, [-0.05, 0.05]); xlim([300, 500]); title('Gradient in v');
subplot(154); imshow(Im, [ 0.00, 0.05]); xlim([300, 500]); title('Gradient magnitude');
subplot(155);
scatter(u, v, 1, theta);
colormap(gca, 'hsv');
box on; axis image;
set(gca, 'YDir', 'reverse');
xlim([300, 500]); 
ylim([0, size(I_rgb,1)]); 
title('Extracted edge points');
c = colorbar('southoutside');
c.Label.String = 'Edge angle (radians)';