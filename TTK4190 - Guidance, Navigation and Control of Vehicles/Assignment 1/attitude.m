% M-script for numerical integration of the attitude dynamics of a rigid 
% body represented by unit quaternions. The MSS m-files must be on your
% Matlab path in order to run the script.
%
% System:                      .
%                              q = T(q)w
%                              .
%                            I w - S(Iw)w = tau
% Control law:
%                            tau = constant
% 
% Definitions:             
%                            I = inertia matrix (3x3)
%                            S(w) = skew-symmetric matrix (3x3)
%                            T(q) = transformation matrix (4x3)
%                            tau = control input (3x1)
%                            w = angular velocity vector (3x1)
%                            q = unit quaternion vector (4x1)
%
% Author:                   2018-08-15 Thor I. Fossen and H�kon H. Helgesen

%% USER INPUTS
h = 0.1;                     % sample time (s)
N  = 4000;                    % number of samples. Should be adjusted

% model parameters
m = 180;
r = 2;
I = m*r^2*eye(3);            % inertia matrix
I_inv = inv(I);
% 1.3
% kp = 2;
% kd = 40;
% 1.5 + 1.6
kp = 20;
kd = 400;
K = [-kp*eye(3), -kd*eye(3)]; % State feedback

T_inv = @(phi, theta, psi) [1, 0, -sin(theta);
    0, cos(phi), cos(theta)*sin(phi);
    0, -sin(phi), cos(theta)*cos(phi)];

% constants
deg2rad = pi/180;   
rad2deg = 180/pi;

phi = -5*deg2rad;            % initial Euler angles
theta = 10*deg2rad;
psi = -20*deg2rad;

q = euler2q(phi,theta,psi);   % transform initial Euler angles to q

w = [0 0 0]';                 % initial angular rates

table = zeros(N+1,14);        % memory allocation

%% FOR-END LOOP
for i = 1:N+1,
%% 1.3
%    t = (i-1)*h;                  % time
%    x = [q(2:end); w];
%    tau = K*x;            % control law
% 
%    [phi,theta,psi] = q2euler(q); % transform q to Euler angles
%    [J,J1,J2] = quatern(q);       % kinematic transformation matrices
%    
%    q_dot = J2*w;                        % quaternion kinematics
%    w_dot = I_inv*(Smtrx(I*w)*w + tau);  % rigid-body kinetics
%    
%    table(i,:) = [t q' phi theta psi w' tau'];  % store data in table
%    
%    q = q + h*q_dot;	             % Euler integration
%    w = w + h*w_dot;
%    
%    q  = q/norm(q);               % unit quaternion normalization
%% 1.5
%    t = (i-1)*h;                  % time
%    
%    phi_desired = 0.0;            % desired Euler angles
%    theta_desired = 15*cos(0.1*t)*deg2rad;
%    psi_desired = 10*sin(0.05*t)*deg2rad;
%    
%    qd = euler2q(phi_desired, theta_desired, psi_desired);
%    qd_conj = [qd(1); -qd(2:end)];
%    
%    q_err = quatprod(qd_conj, q);
%    
%    u = [q_err(2:end); w];
%    tau = K*u;            % control law
% 
%    [phi,theta,psi] = q2euler(q); % transform q to Euler angles
%    [J,J1,J2] = quatern(q);       % kinematic transformation matrices
%    
%    q_dot = J2*w;                        % quaternion kinematics
%    w_dot = I_inv*(Smtrx(I*w)*w + tau);  % rigid-body kinetics
%    
%    table(i,:) = [t q' phi theta psi w' tau'];  % store data in table
%    
%    q = q + h*q_dot;	             % Euler integration
%    w = w + h*w_dot;
%    
%    q  = q/norm(q);               % unit quaternion normalization
%% 1.6   
t = (i-1)*h;                  % time
   
   phi_desired = 0.0;            % desired Euler angles
   theta_desired = 15*cos(0.1*t)*deg2rad;
   psi_desired = 10*sin(0.05*t)*deg2rad;
   
   qd = euler2q(phi_desired, theta_desired, psi_desired);
   qd_conj = [qd(1); -qd(2:end)];
   
   q_err = quatprod(qd_conj, q);
   
   % Big theta dot in (2.39)
   Theta_dot = [0, -0.1*15*sin(0.1*t), 10*0.05*cos(0.05*t)]'*deg2rad;
   wd = T_inv(phi, theta, psi)*Theta_dot;
   w_err = w - wd;
   
   u = [q_err(2:end); w_err];
   tau = K*u;            % control law

   [phi,theta,psi] = q2euler(q); % transform q to Euler angles
   [J,J1,J2] = quatern(q);       % kinematic transformation matrices
   
   q_dot = J2*w;                        % quaternion kinematics
   w_dot = I_inv*(Smtrx(I*w)*w + tau);  % rigid-body kinetics
   
   table(i,:) = [t q' phi theta psi w' tau'];  % store data in table
   
   q = q + h*q_dot;	             % Euler integration
   w = w + h*w_dot;
   
   q  = q/norm(q);               % unit quaternion normalization
end 

%% PLOT FIGURES
t       = table(:,1);  
q       = table(:,2:5); 
phi     = rad2deg*table(:,6);
theta   = rad2deg*table(:,7);
psi     = rad2deg*table(:,8);
w       = rad2deg*table(:,9:11);  
tau     = table(:,12:14);


figure (1); clf;
hold on;
plot(t, phi, 'b');
plot(t, theta, 'r');
plot(t, psi, 'g');
hold off;
grid on;
legend('\phi', '\theta', '\psi');
title('Euler angles');
xlabel('time [s]'); 
ylabel('angle [deg]');

figure (2); clf;
hold on;
plot(t, w(:,1), 'b');
plot(t, w(:,2), 'r');
plot(t, w(:,3), 'g');
hold off;
grid on;
legend('x', 'y', 'z');
title('Angular velocities');
xlabel('time [s]'); 
ylabel('angular rate [deg/s]');

figure (3); clf;
hold on;
plot(t, tau(:,1), 'b');
plot(t, tau(:,2), 'r');
plot(t, tau(:,3), 'g');
hold off;
grid on;
legend('x', 'y', 'z');
title('Control input');
xlabel('time [s]'); 
ylabel('input [Nm]');