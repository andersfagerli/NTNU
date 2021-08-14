path.data = 'Collected data';
path.plots = 'Lab 4 - New/plots/';

%% Problem 6 - Gyro equilibrium no motor
%{
plot.gyro.data = load('task6_gyro_equilbrium_no_motor.mat');

plot.gyro_x.data = plot.gyro.data.ans([1,2],:);
plot.gyro_x.legend = '$\dot{p}_{gyro}$';

plot.gyro_y.data = plot.gyro.data.ans([1,3],:);
plot.gyro_y.legend = '$\dot{e}_{gyro}$';

plot.gyro_z.data = plot.gyro.data.ans([1,4],:);
plot.gyro_z.legend = '$\dot{\lambda}_{gyro}$';

p = Plotex(...
    plot.gyro_x.data(1, :), plot.gyro_x.data(2, :), plot.gyro_x.legend,...
    plot.gyro_y.data(1, :), plot.gyro_y.data(2, :), plot.gyro_y.legend,...
    plot.gyro_z.data(1, :), plot.gyro_z.data(2, :), plot.gyro_z.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$\omega [rad/s]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'gyro_equi_no_motor');
%}

%% Problem 6 - Gyro equilibrium motor
%{
plot.gyro.data = load('task6_gyro_equilbrium.mat');

plot.gyro_x.data = plot.gyro.data.ans([1,2],:);
plot.gyro_x.legend = '$\dot{p}_{gyro}$';

plot.gyro_y.data = plot.gyro.data.ans([1,3],:);
plot.gyro_y.legend = '$\dot{e}_{gyro}$';

plot.gyro_z.data = plot.gyro.data.ans([1,4],:);
plot.gyro_z.legend = '$\dot{\lambda}_{gyro}$';

p = Plotex(...
    plot.gyro_x.data(1, :), plot.gyro_x.data(2, :), plot.gyro_x.legend,...
    plot.gyro_y.data(1, :), plot.gyro_y.data(2, :), plot.gyro_y.legend,...
    plot.gyro_z.data(1, :), plot.gyro_z.data(2, :), plot.gyro_z.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$\omega [rad/s]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'gyro_equi_motor');
%}
%% Problem 6 - Gyro ground
plot.gyro.data = load('task6_gyro_ground.mat');

plot.gyro_x.data = plot.gyro.data.ans([1,2],:);
plot.gyro_x.legend = '$\dot{p}_{gyro}$';

plot.gyro_y.data = plot.gyro.data.ans([1,3],:);
plot.gyro_y.legend = '$\dot{e}_{gyro}$';

plot.gyro_z.data = plot.gyro.data.ans([1,4],:);
plot.gyro_z.legend = '$\dot{\lambda}_{gyro}$';

p = Plotex(...
    plot.gyro_x.data(1, :), plot.gyro_x.data(2, :), plot.gyro_x.legend,...
    plot.gyro_y.data(1, :), plot.gyro_y.data(2, :), plot.gyro_y.legend,...
    plot.gyro_z.data(1, :), plot.gyro_z.data(2, :), plot.gyro_z.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$\omega [rad/s]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'gyro_ground');