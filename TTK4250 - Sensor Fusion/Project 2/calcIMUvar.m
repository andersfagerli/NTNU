clear;
load task_simulation.mat;
dt = mean(diff(timeIMU));
a_est.raw = diff(xtrue(4:6, :)')' ./ dt;
% a_est.filtered = lowpass(a_est.raw, 30, 100);
a_est.time = timeIMU(2:end);
% plot(a_est.time, a_est.raw(3, :)); hold on
plot(timeIMU, zGyro(3, :));
% plot(a_est.time, a_est.filtered);
std(zGyro(3, 35/dt:85/dt))
plot(timeIMU, zAcc(3, :));
% plot(a_est.time, a_est.filtered);
std(zAcc(3, 35/dt:85/dt))