path.data = 'Collected data';
path.plots = 'Lab 5/plots/';

radToDeg = 180/pi;

%% Problem 2 - Encoders sine

plot.encoder.data = load('task2_encoder_sine.mat');

plot.encoder_pitch.data = plot.encoder.data.ans([1,2],:);
plot.encoder_pitch.legend = '$p_{encoder}$';

p = Plotex(...
    plot.encoder_pitch.data(1, :), plot.encoder_pitch.data(2, :)*radToDeg, plot.encoder_pitch.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^\circ]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'pitch_encoder_sine');

%% Problem 2 - Encoders sine

plot.predict.data = load('task2_predicted_sine.mat');

plot.predict_pitch.data = plot.predict.data.ans([1,3],:);
plot.predict_pitch.legend = '$p_{predicted}$';

p = Plotex(...
    plot.predict_pitch.data(1, :), plot.predict_pitch.data(2, :)*radToDeg, plot.predict_pitch.legend,...
    'grid', true,...
    'thick_lines', true,...
    'xlabel', '$t$ [s]',...
    'ylabel', '$p [^\circ]$');

p.plot.plot2pdf('path', path.plots, 'filename', 'pitch_predict_sine');