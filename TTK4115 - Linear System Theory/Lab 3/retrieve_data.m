path.data = 'Collected data';
path.plots = 'Lab 3/plots/';

radToDeg = 180/pi;
reference = 0.3;

%% Problem 3 - Tuning pitch angle

%{
plot.Q_10_1_50.data = load('step_pitchtune_Q_10_1_50.mat');
plot.Q_10_1_50.data = plot.Q_10_1_50.data.ans;
plot.Q_10_1_50.legend = '$q_1 = 10$';


plot.Q_100_1_50.data = load('step_pitchtune_Q_100_1_50.mat');
plot.Q_100_1_50.data = plot.Q_100_1_50.data.ans;
plot.Q_100_1_50.legend = '$q_1 = 100$';


plot.Q_500_1_50.data = load('step_pitchtune_Q_500_1_50');
plot.Q_500_1_50.data = plot.Q_500_1_50.data.ans; 
plot.Q_500_1_50.legend = '$q_1 = 500$';

plot.reference.data = [plot.Q_10_1_50.data(1,:); ones(1,length(plot.Q_10_1_50.data(1,:)))*reference];
plot.reference.legend = '$p_c$';

p = Plotex(...
    plot.Q_10_1_50.data(1, :), plot.Q_10_1_50.data(2, :)*radToDeg, plot.Q_10_1_50.legend,...
    plot.Q_100_1_50.data(1, :), plot.Q_100_1_50.data(2, :)*radToDeg, plot.Q_100_1_50.legend,...
    plot.Q_500_1_50.data(1, :), plot.Q_500_1_50.data(2, :)*radToDeg, plot.Q_500_1_50.legend,...
    plot.reference.data(1,: ), plot.reference.data(2,: )*radToDeg, plot.reference.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'plot_of_pitch_angle');
%}

%% Problem 3 - Tuning R
%{

plot.R_1_0_1.data = load('step_pitchtune_R_1_0_1.mat');
plot.R_1_0_1.data = plot.R_1_0_1.data.ans;
plot.R_1_0_1.legend = '$r_2 = 0.1$';

plot.R_1_1.data = load('step_pitchtune_R_1_1.mat');
plot.R_1_1.data = plot.R_1_1.data.ans;
plot.R_1_1.legend = '$r_2 = 1$';

plot.R_1_10.data = load('step_pitchtune_R_1_10.mat');
plot.R_1_10.data = plot.R_1_10.data.ans;
plot.R_1_10.legend = '$r_2 = 10$';

p = Plotex(...
    plot.R_1_0_1.data(1, :), plot.R_1_0_1.data(2, :)*radToDeg, plot.R_1_0_1.legend,...
    plot.R_1_1.data(1, :), plot.R_1_1.data(2, :)*radToDeg, plot.R_1_1.legend,...
    plot.R_1_10.data(1, :), plot.R_1_10.data(2, :)*radToDeg, plot.R_1_10.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'R_tuning_plot_of_pitch_angle');

%}
%{
plot.R_0_1_1.data = load('step_elevtune_R_0_1_1.mat');
plot.R_0_1_1.data = plot.R_0_1_1.data.ans;
plot.R_0_1_1.legend = '$r_1 = 0.1$';

plot.R_1_1.data = load('step_elevtune_R_1_1.mat');
plot.R_1_1.data = plot.R_1_1.data.ans;
plot.R_1_1.legend = '$r_1 = 1$';

plot.R_10_1.data = load('step_elevtune_R_10_1.mat');
plot.R_10_1.data = plot.R_10_1.data.ans;
plot.R_10_1.legend = '$r_1 = 10$';

p = Plotex(...
    plot.R_0_1_1.data(1, :), plot.R_0_1_1.data(2, :)*radToDeg, plot.R_0_1_1.legend,...
    plot.R_1_1.data(1, :), plot.R_1_1.data(2, :)*radToDeg, plot.R_1_1.legend,...
    plot.R_10_1.data(1, :), plot.R_10_1.data(2, :)*radToDeg, plot.R_10_1.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$e [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'R_tuning_plot_of_elevation_angle');
%}
%% Problem 3 - Effect of F
%{
plot.F_mult_05.data = load('step_pitch_F_mult_05.mat');
plot.F_mult_05.data = plot.F_mult_05.data.ans;
plot.F_mult_05.legend = '$a = 0.5$';

plot.F_mult_1.data = load('step_pitch_F_mult_1.mat');
plot.F_mult_1.data = plot.F_mult_1.data.ans;
plot.F_mult_1.legend = '$a = 1$';

plot.F_mult_2.data = load('step_pitch_F_mult_2.mat');
plot.F_mult_2.data = plot.F_mult_2.data.ans;
plot.F_mult_2.legend = '$a = 2$';

p = Plotex(...
    plot.F_mult_05.data(1, :), plot.F_mult_05.data(2, :)*radToDeg, plot.F_mult_05.legend,...
    plot.F_mult_1.data(1, :), plot.F_mult_1.data(2, :)*radToDeg, plot.F_mult_1.legend,...
    plot.F_mult_2.data(1, :), plot.F_mult_2.data(2, :)*radToDeg, plot.F_mult_2.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'F_pitch_angle');
%}
%% Problem 3 - Integral effect q_4
%{
plot.Q_01_1.data = load('step_pitchtune_integral_effect_Q_01_1.mat');
plot.Q_01_1.data = plot.Q_01_1.data.ans;
plot.Q_01_1.legend = '$q_4 = 0.1$';

plot.Q_1_1.data = load('step_pitchtune_integral_effect_Q_1_1.mat');
plot.Q_1_1.data = plot.Q_1_1.data.ans;
plot.Q_1_1.legend = '$q_4 = 1$';

plot.Q_100_1.data = load('step_pitchtune_integral_effect_Q_100_1.mat');
plot.Q_100_1.data = plot.Q_100_1.data.ans;
plot.Q_100_1.legend = '$q_4 = 100$';

plot.reference.data = [plot.Q_01_1.data(1,:); ones(1,length(plot.Q_01_1.data(1,:)))*reference];
plot.reference.legend = '$p_c$';

p = Plotex(...
    plot.Q_01_1.data(1, :), plot.Q_01_1.data(2, :)*radToDeg, plot.Q_01_1.legend,...
    plot.Q_1_1.data(1, :), plot.Q_1_1.data(2, :)*radToDeg, plot.Q_1_1.legend,...
    plot.Q_100_1.data(1, :), plot.Q_100_1.data(2, :)*radToDeg, plot.Q_100_1.legend,...
    plot.reference.data(1, :), plot.reference.data(2, :)*radToDeg, plot.reference.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'integral_effect_pitch');
%}
%% Problem 3 - Integral effect q_5
%{
plot.Q_1_1.data = load('step_elevtune_integral_effect_Q_1_1.mat');
plot.Q_1_1.data = plot.Q_1_1.data.ans;
plot.Q_1_1.legend = '$q_5 = 1$';

plot.Q_1_100.data = load('step_elevtune_integral_effect_Q_1_100.mat');
plot.Q_1_100.data = plot.Q_1_100.data.ans;
plot.Q_1_100.legend = '$q_5 = 100$';

plot.Q_1_1000.data = load('step_elevtune_integral_effect_Q_1_1000.mat');
plot.Q_1_1000.data = plot.Q_1_1000.data.ans;
plot.Q_1_1000.legend = '$q_5 = 1000$';

p = Plotex(...
    plot.Q_1_1.data(1, :), plot.Q_1_1.data(2, :)*radToDeg, plot.Q_1_1.legend,...
    plot.Q_1_100.data(1, :), plot.Q_1_100.data(2, :)*radToDeg, plot.Q_1_100.legend,...
    plot.Q_1_1000.data(1, :), plot.Q_1_1000.data(2, :)*radToDeg, plot.Q_1_1000.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$e [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'integral_effect_elevation');
%}
%% Problem 3 - Integral effect on F
plot.F_mult_05.data = load('step_pitchtune_integral_effect_Q_1_100_F_mult_0_5.mat');
plot.F_mult_05.data = plot.F_mult_05.data.ans;
plot.F_mult_05.legend = '$a = 0.5$';

plot.F_mult_1.data = load('step_pitchtune_integral_effect_Q_1_100_F_mult_1.mat');
plot.F_mult_1.data = plot.F_mult_1.data.ans;
plot.F_mult_1.legend = '$a = 1$';

plot.F_mult_1_5.data = load('step_pitchtune_integral_effect_Q_1_100_F_mult_1_5.mat');
plot.F_mult_1_5.data = plot.F_mult_1_5.data.ans;
plot.F_mult_1_5.legend = '$a = 1.5$';

p = Plotex(...
    plot.F_mult_05.data(1, :), plot.F_mult_05.data(2, :)*radToDeg, plot.F_mult_05.legend,...
    plot.F_mult_1.data(1, :), plot.F_mult_1.data(2, :)*radToDeg, plot.F_mult_1.legend,...
    plot.F_mult_1_5.data(1, :), plot.F_mult_1_5.data(2, :)*radToDeg, plot.F_mult_1_5.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'F_integral_pitch');