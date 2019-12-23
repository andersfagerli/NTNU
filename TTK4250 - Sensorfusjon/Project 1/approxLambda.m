grid_size = 2500*3000; % Very roughly from plots
m = mean(cellfun(@numel, Z));
lambda = (m - PD) / grid_size;
fprintf('lambda: %e\n', lambda);