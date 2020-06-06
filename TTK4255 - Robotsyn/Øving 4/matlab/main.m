clear;
all_markers        = load('../data/markers.txt');
K                  = load('../data/cameraK.txt');
platform_to_camera = load('../data/pose.txt');
p_model            = load('../data/model.txt');

m = 7; % Number of markers (not all are detected in each frame)

% Initial estimate
yaw   = 11.6*pi/180;
pitch = 28.9*pi/180;
roll  = -0.6*pi/180;

% Task 1
method = @gauss_newton;
last_image = 86;

% Task 2
% method = @levenberg_marquardt;
% last_image = 360;

trajectory = zeros([last_image + 1, 3]);
for image_number=0:last_image
    markers = all_markers(image_number + 1, :)';
    markers = reshape(markers, [3, m])';
    weights = markers(:,1); % weight = 1 if marker was detected or 0 otherwise
    uv = markers(:,2:3);
    if image_number == 0
        r = residuals(K, platform_to_camera, p_model, uv, weights, yaw, pitch, roll);
        disp('Residuals on video0000 are:')
        r
    end
    [yaw, pitch, roll] = gauss_newton(K, platform_to_camera, p_model, uv, weights, yaw, pitch, roll);
    trajectory(image_number + 1, :) = [yaw, pitch, roll];
end

% State estimate provided by the encoder logs
logs = load('../data/logs.txt');
log_time = logs(:,1);
log_yaw = logs(:,2);
log_pitch = logs(:,3);
log_roll = logs(:,4);

video_fps = 16; % Image sequence is 16 images / second
video_time = (0:last_image)/video_fps;

subplot(311);
plot(log_time, log_yaw); hold on;
plot(video_time, trajectory(:,1));
legend('Encoder log', 'Vision estimate');
xlim([0, last_image/video_fps]);
ylim([-1, 1]);
ylabel('Yaw');

subplot(312);
plot(log_time, log_pitch); hold on;
plot(video_time, trajectory(:,2));
xlim([0, last_image/video_fps]);
ylim([-0.25, 0.6]);
ylabel('Pitch');

subplot(313);
plot(log_time, log_roll); hold on;
plot(video_time, trajectory(:,3));
xlim([0, last_image/video_fps]);
ylim([-0.6, 0.6]);
ylabel('Roll');
xlabel('Time (Seconds)');

print('out.png', '-dpng');
