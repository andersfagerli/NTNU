path.data = 'Collected data';
path.plots = 'Lab 2/plots/';
%{
%% Problem 2 - Pole Placement

% Complex stable poles
pole_place.complex.data = load('lab2_pole_placement_complex.mat');
pole_place.complex.data = pole_place.complex.data.ans;
pole_place.complex.legend = 'Complex stable';

% Complex unstable poles
pole_place.complex_unstable.data = load('lab2_pole_placement_complex_unstable.mat');
pole_place.complex_unstable.data = pole_place.complex_unstable.data.ans;
pole_place.complex_unstable.legend = 'Complex unstable';

% Real negative overlapping poles
pole_place.overlapping.data = load('lab2_pole_placement_overlapping.mat');
pole_place.overlapping.data = pole_place.overlapping.data.ans; 
pole_place.overlapping.legend = 'Real negative overlapping';

% Real non-overlapping poles
pole_place.real.data = load('lab2_pole_placement_real.mat');
pole_place.real.data = pole_place.real.data.ans;
pole_place.real.legend = 'Real negative non-overlapping';

p = Plotex(...
    pole_place.complex.data(1, :), pole_place.complex.data(2, :), pole_place.complex.legend,...
    pole_place.complex_unstable.data(1, :), pole_place.complex_unstable.data(2, :), pole_place.complex_unstable.legend,...
    pole_place.overlapping.data(1, :), pole_place.overlapping.data(2, :), pole_place.overlapping.legend,...
    pole_place.real.data(1,:), pole_place.real.data(2,:), pole_place.real.legend, ...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'plot_of_different_pole_placements_part2_problem2');
%}

%% Problem 3 - Harmonic oscillator
% zeta = 0

pole_place.w_0_531.data = load('lab2_harmonic_oscillator_w_0_531.mat');
pole_place.w_0_531.data = pole_place.w_0_531.data.ans;
pole_place.w_0_531.legend = '$w_{0} = 0.531$';

pole_place.w_0_2375.data = load('lab2_harmonic_oscillator_w_0_2375.mat');
pole_place.w_0_2375.data = pole_place.w_0_2375.data.ans;
pole_place.w_0_2375.legend = '$w_{0} = 0.2375$';

pole_place.w_1_0621.data = load('lab2_harmonic_oscillator_w_1_0621.mat');
pole_place.w_1_0621.data = pole_place.w_1_0621.data.ans; 
pole_place.w_1_0621.legend = '$w_{0} = 1.0621$';

p = Plotex(...
    pole_place.w_0_2375.data(1, :), pole_place.w_0_2375.data(2, :), pole_place.w_0_2375.legend,...
    pole_place.w_0_531.data(1, :), pole_place.w_0_531.data(2, :), pole_place.w_0_531.legend,...
    pole_place.w_1_0621.data(1, :), pole_place.w_1_0621.data(2, :), pole_place.w_1_0621.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'harmonic_oscillator_w0');

% omega = const
%{
pole_place.zeta_0_5.data = load('lab2_harmonic_oscillator_zeta_0_5.mat');
pole_place.zeta_0_5.data = pole_place.zeta_0_5.data.ans;
pole_place.zeta_0_5.data(1,:) = pole_place.zeta_0_5.data(1,:)+5;
pole_place.zeta_0_5.legend = '$\zeta = 0.5$';

pole_place.zeta_0_7.data = load('lab2_harmonic_oscillator_zeta_0_7.mat');
pole_place.zeta_0_7.data = pole_place.zeta_0_7.data.ans;
pole_place.zeta_0_7.legend = '$\zeta = 0.7$';

pole_place.zeta_1.data = load('lab2_harmonic_oscillator_zeta_1.mat');
pole_place.zeta_1.data = pole_place.zeta_1.data.ans; 
pole_place.zeta_1.legend = '$\zeta = 1$';

pole_place.zeta_1_5.data = load('lab2_harmonic_oscillator_zeta_1_5.mat');
pole_place.zeta_1_5.data = pole_place.zeta_1_5.data.ans; 
pole_place.zeta_1_5.legend = '$\zeta = 1.5$';

p = Plotex(...
    pole_place.zeta_0_5.data(1, :), pole_place.zeta_0_5.data(2, :), pole_place.zeta_0_5.legend,...
    pole_place.zeta_0_7.data(1, :), pole_place.zeta_0_7.data(2, :), pole_place.zeta_0_7.legend,...
    pole_place.zeta_1.data(1, :), pole_place.zeta_1.data(2, :), pole_place.zeta_1.legend,...
    pole_place.zeta_1_5.data(1, :), pole_place.zeta_1_5.data(2, :), pole_place.zeta_1_5.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^{\circ}]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'harmonic_oscillator_zeta');
%}
