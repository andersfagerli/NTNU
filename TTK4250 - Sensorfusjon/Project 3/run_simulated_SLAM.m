load simulatedSLAM;
K = numel(z);
%%
Q = diag([0.5 0.5 3*pi/180].^2);
R = diag([0.06 2*pi/180].^2);
  
doAsso = true;
JCBBalphas = [0.05, (1-chi2cdf(3^2,2))]; % first is for joint compatibility, second is individual
slam = EKFSLAM(Q, R, doAsso, JCBBalphas);

% allocate
xpred = cell(1, K);
Ppred = cell(1, K);
xhat = cell(1, K);
Phat = cell(1, K);
a = cell(1, K);

% init
xpred{1} = poseGT(:,1); % we start at the correct position for reference
Ppred{1} = zeros(3, 3); % we also say that we are 100% sure about that

pose_errors = zeros(3, K);
figure(10); clf;
axAsso = gca;
N = K;
alpha = 0.05;
doAssoPlot = true; % set to true to se the associations that are done
for k = 1:N
    disp(k);
    [xhat{k}, Phat{k}, NIS(k), a{k}] =  slam.update(xpred{k}, Ppred{k}, z{k});
    if k < K
        [xpred{k + 1}, Ppred{k + 1}] = slam.predict(xhat{k}, Phat{k}, odometry(:, k));
    end
    
    NEES(k) = (xhat{k}(1:3) - poseGT(:,k))' / Phat{k}(1:3,1:3) * (xhat{k}(1:3) - poseGT(:,k)); % NEES pose
    NIS_CI(:,k) = chi2inv([alpha/2, 1-alpha/2], length(z{k}))'; % CI bounds for each measurement vector
    pose_errors(:,k) = (xhat{k}(1:3) - poseGT(:,k))';
    % checks
    if size(xhat{k},1) ~= size(Phat{k},1)
        error('dimensions of mean and covariance do not match')
    end
    
%     if doAssoPlot && k > 1 %&& any(a{k} == 0) % uncoment last part to only see new creations
%         cla(axAsso); hold on;grid  on;
%         zpred = reshape(slam.h(xpred{k}), 2, []);
%         scatter(axAsso, z{k}(1, :), z{k}(2, :));
%         scatter(axAsso, zpred(1, :), zpred(2, :));
%         plot(axAsso, [z{k}(1, a{k}>0); zpred(1, a{k}(a{k}>0))], [z{k}(2, a{k}>0); zpred(2, a{k}(a{k}>0))], 'r', 'linewidth', 2)
%         
%         legend(axAsso, 'z', 'zbar', 'a')
%         title(axAsso, sprintf('k = %d: %s', k, sprintf('%d, ',a{k})));
%         %pause();
%     end
    display(k)
end

%% RMSE
pose_rmse = sqrt(mean(pose_errors.^2,2));
pose_abs_errors = sqrt(pose_errors.^2);

figure(2); clf; hold on;
subplot(2,1,1); hold on;
plot(pose_abs_errors(1,:)')
plot(pose_abs_errors(2,:)')
ylabel('Position error [m]')
grid on;
legend(sprintf('RMSE_x (%.3g)',pose_rmse(1)),...
    sprintf('RMSE_y (%.3g)', pose_rmse(2)));

subplot(2,1,2);
plot(pose_abs_errors(3,:)')
ylabel('Heading error [rad]');
title(sprintf('RMSE: %.3g', pose_rmse(3)));
grid on;

%% plotting
figure(3);
k = N;
clf;
%subplot(1,2,1);
hold on;

scatter(landmarks(1,:), landmarks(2,:), 'r^')
scatter(xhat{k}(4:2:end), xhat{k}(5:2:end), 'b.')

lh1 = plot(poseGT(1, 1:k), poseGT(2,1:k), 'r', 'DisplayName', 'gt');
lh2 = plot(cellfun(@(x) x(1), xhat), cellfun(@(x) x(2), xhat), 'b', 'DisplayName', 'est');

el = ellipse(xhat{k}(1:2),Phat{k}(1:2,1:2),5,200);
plot(el(1,:),el(2,:),'b');

for ii=1:((size(Phat{k}, 1)-3)/2)
   rI = squeeze(Phat{k}(3+[1,2]+(ii-1)*2,3+[1,2]+(ii-1)*2));
   el = ellipse(xhat{k}(3 + (1:2) + (ii-1)*2),rI,5,200);
   plot(el(1,:),el(2,:),'b');
end

axis equal;
title('Estimates of pose and landmarks over the simulation');
legend([lh1, lh2])
grid on;

%% NIS, using varying degrees of freedom
% alpha = 0.05;
% ANIS = mean(NIS)
% ACI = chi2inv([alpha/2; 1 - alpha/2], 1)/N % NOT CORRECT NOW
% CI = chi2inv([alpha/2; 1 - alpha/2], 1); % NOT CORRECT NOW
% warning('These consistency intervals have wrong degrees of freedom')

figure(5); clf;
hold on;
plot(1:N, NIS(1:N));
insideCI = mean((NIS_CI(1,end-1) < NIS(2:end) .* (NIS(2:end) <= NIS_CI(2,end-1))))*100
plot(1:N, NIS_CI,'r');

title(sprintf('NIS over time, with %0.1f%% inside %0.1f%% CI', insideCI, (1-alpha)*100));
grid on;
ylabel('NIS');
xlabel('timestep');

%% NEES (pose)
alpha = 0.05;
ANEES = mean(NEES);
CI3 = chi2inv([alpha/2; 1-alpha/2], 3);

figure(6); clf;
hold on;
plot(1:N, NEES(1:N));
insideCI = mean((CI3(1) < NEES) .* (NEES <= CI3(2)))*100;
plot([1, N], (CI3*ones(1, 2))','r--');
title(sprintf('NEES over time, with %0.1f%% inside %0.1f%% CI', insideCI, (1-alpha)*100));
grid on;
ylabel('NEES');
xlabel('timestep');


%% run a movie
% pauseTime = 0.05;
% fig = figure(4);
% ax = gca;
% for k = 1:N
%     cla(ax); hold on;
%     scatter(ax, landmarks(1,:), landmarks(2,:), 'r^')
%     scatter(ax, xhat{k}(4:2:end), xhat{k}(5:2:end), 'b*')
%     plot(ax, poseGT(1, 1:k), poseGT(2,1:k), 'r-o','markerindices',10:10:k);
%     plot(ax, cellfun(@(x) x(1), xhat(1:k)), cellfun(@(x) x(2), xhat(1:k)), 'b-o','markerindices',10:10:k);
%     
%     if k > 1 % singular cov at k = 1
%         el = ellipse(xhat{k}(1:2),Phat{k}(1:2,1:2),5,200);
%         plot(ax,el(1,:),el(2,:),'b');
%     end
%     
%     for ii=1:((size(Phat{k}, 1)-3)/2)
%        rI = squeeze(Phat{k}(3+[1,2]+(ii-1)*2,3+[1,2]+(ii-1)*2)); 
%        el = ellipse(xhat{k}(3 + (1:2) + (ii-1)*2),rI,5,200);
%        plot(ax, el(1,:),el(2,:),'b');
%     end
%     
%     title(ax, sprintf('k = %d',k))
%     grid(ax, 'on');
%     pause(pauseTime);
% end
        