clear; close all; clc;
%% data loading
addpath('./victoria_park');
load('aa3_dr.mat');
load('aa3_lsr2.mat');
load('aa3_gpsx.mat');
timeOdo = time/1000; clear time;
timeLsr = double(TLsr)/1000; clear TLsr;
timeGps = timeGps/1000;
K = numel(timeOdo);
mK = numel(timeLsr);

%% Parameters
% the car parameters
car.L = 2.83; % axel distance
car.H = 0.76; % center to wheel encoder
car.a = 0.95; % laser distance in front of first axel
car.b = 0.5; % laser distance to the left of center

% the SLAM parameters
sigmas = [4.2e-2 , 4.2e-2, 2e-2]; % what is x, y, a car has more process noise in forward direction
CorrCoeff = [1, 0, 0; 0, 1, 0.9; 0, 0.9, 1];
Q = diag(sigmas) * [1, 0, 0; 0, 1, 0.9; 0, 0.9, 1] * diag(sigmas); % (a bit at least) emprically found, feel free to change

R = diag([1e-2, 1.1e-2]);

JCBBalphas = [1e-5, 1e-3]; % fiwWrst is for joint compatibility, second is individual 
sensorOffset = [car.a + car.L; car.b];
slam = EKFSLAM(Q, R, true, JCBBalphas, sensorOffset);

% allocate
xupd = zeros(3, mK);
a = cell(1, mK);

% initialize TWEAK THESE TO BETTER BE ABLE TO COMPARE TO GPS
eta = [Lo_m(1); La_m(2); 30 * pi /180]; % set the start to be relatable to GPS. 
P = zeros(3,3); % we say that we start knowing where we are in our own local coordinates

mk = 2; % first seems to be a bit off in timing
t = timeOdo(1);
tic
N = 15000;


doPlot = true;
figure(1); clf;  hold on; grid on; axis equal;
ax = gca;
% cheeper to update plot data than to create new plot objects
lhPose = plot(ax, eta(1), eta(2), 'k');
shLmk = scatter(ax, nan, nan, 'rx');
shZ = scatter(ax, nan, nan, 'bo');
th = title(ax, 'start');

alpha = 0.05;
theta = -pi/12;
gpsCounter = 1;
count = 1;
for k = 1:N
    if mk < mK && timeLsr(mk) <= timeOdo(k+1)
        dt = timeLsr(mk) - t;
        if  dt >= 0
            t = timeLsr(mk);
            odo = odometry(speed(k + 1), steering(k + 1), dt, car);
            [eta, P] = slam.predict(eta, P, odo);
        else
            error('negative time increment...')
        end
        z = detectTreesI16(LASER(mk,:));
        [eta, P, NIS(k), a{k}] = slam.update(eta, P, z);
        xupd(:, mk) = eta(1:3); 
        NIS_CI(:, k) = chi2inv([alpha/2, 1-alpha/2], size(z, 2))'; % CI bounds for each measurement vector

        %% NIS calc
        len_vk(k) = 2 * nnz(a{k});
        if len_vk(k) > 0
            
            NISS(count) = NIS(k);

            alpha = 0.05;

            CI(:,count) = chi2inv([alpha/2; 1 - alpha/2], len_vk(k));
            CInormalized(:,count) = CI(:,count) / len_vk(k);
            count = count +1;
        end
        %% POS RMSE
       if timeLsr(mk) >=  timeGps(gpsCounter)
           gpsXY = [Lo_m(gpsCounter); La_m(gpsCounter)];
           gpsTrue(:,gpsCounter) = rotmat2d(theta)*gpsXY;
           xnew(:,gpsCounter) = xupd(1:2,mk);
           error(:,gpsCounter) = xupd(1:2,mk)- gpsTrue(:,gpsCounter);
           NEES(:,gpsCounter) = error(:,gpsCounter)'/(P(1:2,1:2)+4*ones(2))*error(:,gpsCounter);
           gpsCounter = gpsCounter + 1;
       end
       mk = mk + 1;
        %{
        if doPlot
            lhPose.XData = [lhPose.XData, eta(1)];
            lhPose.YData = [lhPose.YData, eta(2)];
            shLmk.XData = eta(4:2:end);
            shLmk.YData = eta(5:2:end);
            if ~isempty(z)
                zinmap = rotmat2d(eta(3)) * (z(1,:).*[cos(z(2,:)); sin(z(2,:))] + slam.sensOffset) + eta(1:2);
                shZ.XData = zinmap(1,:);
                shZ.YData = zinmap(2,:);
            end

            th.String = sprintf('step %d, laser scan %d, landmarks %d, measurements %d, num new = %d', k, mk, (size(eta,1) - 3)/2, size(z,2), nnz(a{k} == 0));
            drawnow;
            pause(0.01)
        end
        %}
    end
    
    if k < K
        dt = timeOdo(k+1) - t;
        t = timeOdo(k+1);
        odo = odometry(speed(k+1), steering(k+1), dt, car);
        [eta, P] = slam.predict(eta, P, odo);
    end
    if mod(k, 50) == 0
        toc
        disp(k)
        tic
    end
end

%% 
figure(2); clf;  hold on; grid on; axis equal;
plot(xupd(1, 1:(mk-1)), xupd(2, 1:(mk-1)))
scatter(Lo_m(timeGps < timeOdo(N)), La_m(timeGps < timeOdo(N)), '.')
scatter(eta(4:2:end), eta(5:2:end), 'rx');


% what can we do with consistency..? divide by the number of associated
% measurements? 


%% NEES
theta = -0.7*pi/12;
trans = [10;-13];
gpsCounter = 1;
mk = 2;

for k = 1:N
    if mk < mK && timeLsr(mk) <= timeOdo(k+1)
        dt = timeLsr(mk) - t;
        if  dt >= 0
            t = timeLsr(mk);
        else
            error('negative time increment...')
        end


        


       if timeLsr(mk) >=  timeGps(gpsCounter)
           gpsXY = [Lo_m(gpsCounter); La_m(gpsCounter)];
           gpsTrue(:,gpsCounter) = rotmat2d(theta)*gpsXY + trans;
           xnew(:,gpsCounter) = xupd(1:2,mk);
           error(:,gpsCounter) = xupd(1:2,mk)- gpsTrue(:,gpsCounter);
           gpsCounter = gpsCounter + 1;
       end
       mk = mk + 1;
    end
    
    if k < K
        dt = timeOdo(k+1) - t;
        t = timeOdo(k+1);
    end

end


figure(5); clf;  hold on; grid on; axis equal;
scatter(xnew(1,:), xnew(2,:)); hold on;
scatter(gpsTrue(1,:), gpsTrue(2,:), '.')

posRMSE = sqrt(mean(sum(error(:,:).^2,1)))
title(sprintf('PosRMSE:%0.2f%', posRMSE));

%%

figure(10); clf;
hold on;
plot(1:1754, NISS(1:1754));
insideCI = mean((CInormalized(1,:) < NISS) .* (NISS <= CInormalized(2,:)))*100;
plot(CInormalized(1,:),'r--'); hold on;
plot(CInormalized(2,:),'r--'); hold on;

title(sprintf('NIS over time, with %0.1f%% inside %0.1f%% CI', insideCI, (1-alpha)*100));
grid on;
ylabel('NIS');
xlabel('timestep');


%% 
%{
for k = 1:14992
    len_vk(k) = 2 * nnz(a{k});
    CI(:,k) = chi2inv([alpha/2; 1 - alpha/2], len_vk(k));
    CInormalized(:,k) = CI(:,k) / len_vk(k);
end

figure(10); clf;
hold on;
plot(1:14992, NIS(1:14992));
insideCI = mean((CInormalized(1,:) < NIS) .* (NIS <= CInormalized(2,:)))*100;
plot(CInormalized(1,:),'r--'); hold on;
plot(CInormalized(2,:),'r--'); hold on;

title(sprintf('NIS over time, with %0.1f%% inside %0.1f%% CI', insideCI, (1-alpha)*100));
grid on;
ylabel('NIS');
%}


%%
figure(7); clf;

plot(NEES); grid on; hold on;

ciNEES = (chi2inv([0.025, 0.975], 2))/2;
inCI = sum((NEES >= ciNEES(1)) .* (NEES <= ciNEES(2)))/1077 * 100;
plot([1,1077], repmat(ciNEES',[1,2])','r--')
text(104, -5, sprintf('%.2f%% inside CI', inCI),'Rotation',90);

title(sprintf('NEES over time, with %0.1f%% inside %0.1f%% CI', inCI, (1-alpha)*100));
grid on;
ylabel('NEES');
xlabel('timestep');
% clear;
% close all;
% 
% %% data loading
% addpath('./victoria_park');
% load('aa3_dr.mat');
% load('aa3_lsr2.mat');
% load('aa3_gpsx.mat');
% timeOdo = time/1000; clear time;
% timeLsr = double(TLsr)/1000; clear TLsr;
% timeGps = timeGps/1000;
% K = numel(timeOdo);
% mK = numel(timeLsr);
% mGps = numel(timeGps);
% 
% %% Parameters
% % the car parameters
% car.L = 2.83; % axel distance
% car.H = 0.76; % center to wheel encoder
% car.a = 0.95; % laser distance in front of first axel
% car.b = 0.5; % laser distance to the left of center
% 
% % the SLAM parameters
% sigmas = [4e-2, 4e-2, 2e-2]; %[0.05, 0.02, deg2rad(2)];
% % sigmas = [4.5e-1 , 4.5e-1, 5e-2];
% CorrCoeff = [1, 0, 0; 0, 1, 0.9; 0, 0.9, 1];
% Q = diag(sigmas) * [1, 0, 0; 0, 1, 0.9; 0, 0.9, 1] * diag(sigmas); % (a bit at least) emprically found, feel free to change
% 
% % Range and bearing
% R = diag( [1e-2, 1e-2] );%diag([0.01, deg2rad(2)].^2);
% % R = diag([5.5e-2, 5.5e-2].^2);
% 
% JCBBalphas = [1e-5, 1e-3]; %[0.01, 1 - chi2cdf(3^2, 2)]; % first is for joint compatibility, second is individual 
% sensorOffset = [car.a + car.L; car.b];
% slam = EKFSLAM(Q, R, true, JCBBalphas, sensorOffset);
% 
% % allocate
% xupd = zeros(3, mK);
% a = cell(1, mK);
% 
% % offsets
% x_offset = 0;%-1;
% y_offset = 0;%0.0461;
% heading = 30;%35.5;
% 
% theta = -0.7*pi/12;
% trans = [10;-13];
% gpsCounter = 1;
% 
% % initialize TWEAK THESE TO BETTER BE ABLE TO COMPARE TO GPS
% eta = [Lo_m(1) + x_offset; La_m(1) + y_offset; heading * pi /180]; % set the start to be relatable to GPS. 
% P = zeros(3,3); % we say that we start knowing where we are in our own local coordinates
% 
% mk = 2; % first seems to be a bit off in timing
% mgps = 2; % first seems to be a bit off in timing
% t = timeOdo(1);
% %tic
% N = 15000;
% 
% tGPS = 1;
% 
% % doPlot = false;
% % figure(1); clf;  hold on; grid on; axis equal;
% % ax = gca;
% % % cheaper to update plot data than to create new plot objects
% % lhPose = plot(ax, eta(1), eta(2), 'k');
% % shLmk = scatter(ax, nan, nan, 'rx');
% % shZ = scatter(ax, nan, nan, 'bo');
% % th = title(ax, 'start');
% 
% alpha = 0.05;
% %NEES = zeros(N, 1);
% 
% for k = 1:N
%     disp(k)
%     if mk < mK && timeLsr(mk) <= timeOdo(k+1)
%         dt = timeLsr(mk) - t;
% %         if  dt >= 0
%             t = timeLsr(mk);
%             odo = odometry(speed(k + 1), steering(k + 1), dt, car);
%             [eta, P] = slam.predict(eta, P, odo);
% %         else
% %             error('negative time increment...')
% %         end
%         z = detectTreesI16(LASER(mk,:));
%         [eta, P, NIS(k), a{k}] = slam.update(eta, P, z);
%         xupd(:, mk) = eta(1:3); 
%         NIS_CI(:, k) = chi2inv([alpha/2, 1-alpha/2], size(z, 2))'; % CI bounds for each measurement vector
%         mk = mk + 1;
% %         if doPlot
% %             lhPose.XData = [lhPose.XData, eta(1)];
% %             lhPose.YData = [lhPose.YData, eta(2)];
% %             shLmk.XData = eta(4:2:end);
% %             shLmk.YData = eta(5:2:end);
% %             if ~isempty(z)
% %                 zinmap = rotmat2d(eta(3)) * (z(1,:).*[cos(z(2,:)); sin(z(2,:))] + slam.sensOffset) + eta(1:2);
% %                 shZ.XData = zinmap(1,:);
% %                 shZ.YData = zinmap(2,:);
% %             end
% % 
% %             th.String = sprintf('step %d, laser scan %d, landmarks %d, measurements %d, num new = %d', k, mk, (size(eta,1) - 3)/2, size(z,2), nnz(a{k} == 0));
% %             drawnow;
% %             pause(0.01)
% %         end
%         %poseGPS = [Lo_m(k); La_m(k+1)];
%         %NEES(k) = ((xupd(1:2, k) - poseGPS)' / P(1:2, 1:2)) * (xupd(1:2, k) - poseGPS); % NEES pose
%     end
%     
% %     if gpsCounter < mGps && timeGps(gpsCounter) <= timeOdo(k+1)
% % %             gpsXY = [Lo_m(mgps); La_m(mgps)];
% % %             poseGPS = rotmat2d(theta)*gpsXY + trans;
% % %             NEES(mk) = ((xupd(1:2, mk) - poseGPS)' / P(1:2, 1:2)) * (xupd(1:2, mk) - poseGPS); % NEES pose
% % %             posRMSE(:, mgps) = xupd(1:2, mk) - poseGPS;
% % %             mgps = mgps + 1;
% %            gpsXY = [Lo_m(gpsCounter); La_m(gpsCounter)];
% %            gpsTrue(:,gpsCounter) = rotmat2d(theta)*gpsXY + trans;
% %            xnew(:,gpsCounter) = xupd(1:2,mk);
% %            error(:,gpsCounter) = xupd(1:2,mk)- gpsTrue(:,gpsCounter);
% %            gpsCounter = gpsCounter + 1;
% %     end   
% %     
%     if k < K
%         dt = timeOdo(k+1) - t;
%         t = timeOdo(k+1);
%         odo = odometry(speed(k+1), steering(k+1), dt, car);
%         [eta, P] = slam.predict(eta, P, odo);
%     end
% %     if mod(k, 50) == 0
% %         toc
% %         disp(k)
% %         tic
% %     end
%     
%     disp(k)
% end
% 
% %% 
% figure(2); clf;  hold on; grid on; axis equal;
% plot(xupd(1, 1:(mk-1)), xupd(2, 1:(mk-1)))
% scatter(Lo_m(timeGps < timeOdo(N)), La_m(timeGps < timeOdo(N)), '.')
% scatter(eta(4:2:end), eta(5:2:end), 'rx');
% 
% % what can we do with consistency..? divide by the number of associated
% % measurements? 
figure(3); clf;
hold on;
NIS = NIS(NIS > 0);
NIS_CI = [NIS_CI(1, NIS_CI(1, :) > 0);
          NIS_CI(2, NIS_CI(2, :) > 0)];
NIS_CI = [NIS_CI(1, 1:size(NIS, 2));
          NIS_CI(2, 1:size(NIS, 2))];

% NIS_CI = NIS_CI([1, NIS_CI(1, :) > 0; 2, NIS_CI(2, :) > 0;]);
insideCI = mean( ( NIS_CI(1,:) < NIS ) .* ( NIS <= NIS_CI(2,:) ) ) * 100;

plot(NIS_CI(1, :),'r');
plot(NIS_CI(2, :),'r');
plot(NIS);
% 
title(sprintf('NIS over time, with %0.1f%% inside %0.1f%% CI', insideCI, (1-alpha)*100));
grid on;
ylabel('NIS');
xlabel('timestep');
% 
% %% NEES (pose)
% % alpha = 0.05;
% % NEES = NEES(NEES > 0);
% % ANEES = mean(NEES);
% % CI3 = chi2inv([alpha/2; 1-alpha/2], 3);
% % NEES = NEES(NEES > 0);
% % N = size(NEES, 2);
% % 
% % figure(4); clf;
% % hold on;
% % plot(1:N, NEES(1:N));
% % insideCI = mean((CI3(1) < NEES) .* (NEES <= CI3(2)))*100;
% % plot([1, N], (CI3*ones(1, 2))','r--');
% % title(sprintf('NEES over time, with %0.1f%% inside %0.1f%% CI', insideCI, (1-alpha)*100));
% % grid on;
% % ylabel('NEES');
% % xlabel('timestep');
% 
% %% RMSE
% % pose_errors = posRMSE;
% % pose_rmse = sqrt(mean(pose_errors.^2, 2));
% % pose_rmse = pose_rmse(pose_rmse > 0);
% % pose_abs_errors = sqrt(pose_errors.^2);
% % pose_abs_errors = pose_abs_errors(pose_abs_errors > 0);
% % 
% % figure(5); clf; hold on;
% % plot(pose_abs_errors(1,:)')
% % plot(pose_abs_errors(2,:)')
% % ylabel('Position error [m]')
% % grid on;
% % legend(sprintf('RMSE_x (%.3g)',pose_rmse(1)),...
% %     sprintf('RMSE_y (%.3g)', pose_rmse(2)));
% 
% %% NEES
% theta = pi/6;
% gpsCounter = 1;
% deltaT = timeLsr(1756)/1756;
% for i = 1:1756
%    if timeLsr(i) >=  timeGps(gpsCounter)
%        gpsXY = [Lo_m(gpsCounter); La_m(gpsCounter)];
%        gpsTrue(:,gpsCounter) = rotmat2d(theta)*gpsXY;
%        xnew(:,gpsCounter) = xupd(1:2,i);
%        error(:,gpsCounter) = xupd(1:2,i)- gpsTrue(:,gpsCounter);
%        gpsCounter = gpsCounter + 1;
%    end
% end
% 
% figure(5); clf;  hold on; grid on; axis equal;
% plot(xnew(1,:), xnew(2,:)); hold on;
% scatter(gpsTrue(1,:), gpsTrue(2,:), '.')
