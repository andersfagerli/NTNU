% TTK4135 - Helicopter lab
% Hints/template for problem 2.
% Updated spring 2018, Andreas L. Fl�ten

%% Initialization and model definition
init08; % Change this to the init file corresponding to your helicopter

% Discrete time system model. x = [lambda r p p_dot]'
delta_t	= 0.25; % sampling time

Ac = [0, 1,         0,         0,         0,         0;
      0, 0,      -K_2,         0,         0,         0;
      0, 0,         0,         1,         0,         0;
      0, 0, -K_1*K_pp, -K_1*K_pd,         0,         0;
      0, 0,         0,         0,         0,         1;
      0, 0,         0,         0, -K_3*K_ep, -K_3*K_ed];
  
Bc = [       0,        0;
             0,        0;
             0,        0;
      K_1*K_pp,        0;
             0,        0;
             0, K_3*K_ep];

Ad = Ac*delta_t  + eye(size(Ac,1));
Bd = Bc*delta_t;


% Number of states and inputs
mx = size(Ad,2); % Number of states (number of columns in A)
mu = size(Bd,2); % Number of inputs(number of columns in B)

% Initial values
x1_0 = pi;                              % Lambda
x2_0 = 0;                               % r
x3_0 = 0;                               % p
x4_0 = 0;                               % p_dot
x5_0 = 0;                               % e
x6_0 = 0;                               % e_dot
x0 = [x1_0 x2_0 x3_0 x4_0 x5_0 x6_0]';  % Initial values

% Time horizon and initialization
N  = 40;                               % Time horizon for states
M  = N;                                 % Time horizon for inputs
z  = zeros(N*mx+M*mu,1);                % Initialize z for the whole horizon
z0 = z;                                 % Initial value for optimization

% Bounds
ul 	    = [-30*pi/180; -inf];           % Lower bound on control
uu 	    = [30*pi/180; inf];             % Upper bound on control

xl      = -Inf*ones(mx,1);              % Lower bound on states (no bound)
xu      = Inf*ones(mx,1);               % Upper bound on states (no bound)
xl(3)   = ul(1);                           % Lower bound on state x3
xu(3)   = uu(1);                           % Upper bound on state x3

% Generate constraints on measurements and inputs
[vlb,vub]       = gen_constraints(N, M, xl, xu, ul, uu); % hint: gen_constraints
vlb(N*mx+M*mu)  = 0;                    % We want the last input to be zero
vub(N*mx+M*mu)  = 0;                    % We want the last input to be zero

% Generate the matrix G and the vector c (objecitve function weights in the QP problem) 
G1 = zeros(mx,mx);
G1(1,1) = 2;                           % Weight on state x1
G1(2,2) = 0;                           % Weight on state x2
G1(3,3) = 0;                           % Weight on state x3
G1(4,4) = 0;                           % Weight on state x4
G1(5,5) = 0;                           % Weight on state x5
G1(6,6) = 0;                           % Weight on state x6
P1 = 2*diag([1, 0.1]);                   % Weight on input
G = gen_q(G1, P1, N, M);               % Generate G, hint: gen_q
c = [];                                % Generate c, this is the linear constant term in the QP

% LQR
Q = diag([10 1 1 1 1 1]); %[travel, travel_rate, pitch, pitch_rate]
R = [1 0;
     0 1];
K = dlqr(Ad, Bd, Q, R);

%% Generate system matrixes for linear model
Aeq = gen_aeq(Ad, Bd, N, mx, mu);      % Generate A, hint: gen_aeq
beq = zeros(size(Aeq,1),1);            % Generate b
beq(1:6) = Ad*x0;

%% Solve QP problem with linear model
fun = @(z) 0.5*z'*G*z;
options = optimoptions('fmincon');
options.MaxFunEvals = 40000;
tic
[z,lambda] = fmincon(fun, z0, [], [], Aeq, beq, vlb, vub, @nonlincon, options); % hint: quadprog. Type 'doc quadprog' for more info 
t1=toc;

% Calculate objective value
phi1 = 0.0;
PhiOut = zeros(N*mx+M*mu,1);
for i=1:N*mx+M*mu
  phi1=phi1+G(i,i)*z(i)*z(i);
  PhiOut(i) = phi1;
end

%% Extract control inputs and states
u  = z(N*mx+1:N*mx+M*mu);% Control input from solution
u1 = u(1:mu:N*mu);
u2 = u(2:mu:N*mu);
x1 = [x0(1);z(1:mx:N*mx)];              % State x1 from solution
x2 = [x0(2);z(2:mx:N*mx)];              % State x2 from solution
x3 = [x0(3);z(3:mx:N*mx)];              % State x3 from solution
x4 = [x0(4);z(4:mx:N*mx)];              % State x4 from solution
x5 = [x0(5);z(5:mx:N*mx)];              % state x5 from solution
x6 = [x0(6);z(6:mx:N*mx)];              % state x6 from solution

num_variables = 10/delta_t;
zero_padding = zeros(num_variables,1);
unit_padding  = ones(num_variables,1);

u1   = [zero_padding; u1; zero_padding; 0];
u2   = [zero_padding; u2; zero_padding; 0];
x1  = [pi*unit_padding; x1; zero_padding];
x2  = [zero_padding; x2; zero_padding];
x3  = [zero_padding; x3; zero_padding];
x4  = [zero_padding; x4; zero_padding];
x5  = [zero_padding; x5; zero_padding];
x6  = [zero_padding; x6; zero_padding];


%% Plotting
t = 0:delta_t:delta_t*(length(u1)-1);

u1_opt = [t', u1];
u2_opt = [t', u2];
x1_opt = [t', x1];
x2_opt = [t', x2];
x3_opt = [t', x3];
x4_opt = [t', x4];
x5_opt = [t', x5];
x6_opt = [t', x6];

u_opt = [t', u1, u2];
x_opt = [t', x1,x2,x3,x4,x5,x6];

figure(2)
subplot(611)
plot(t,x1,'m',t,x1,'mo'),grid
ylabel('lambda')
subplot(612)
plot(t,x2,'m',t,x2','mo'),grid
ylabel('r')
subplot(613)
plot(t,x3,'m',t,x3,'mo'),grid
ylabel('p')
subplot(614)
plot(t,x4,'m',t,x4','mo'),grid
xlabel('tid (s)'),ylabel('pdot')
subplot(615)
plot(t,x5,'m',t,x5','mo'),grid
xlabel('tid (s)'),ylabel('e')
subplot(616)
plot(t,x6,'m',t,x6','mo'),grid
xlabel('tid (s)'),ylabel('edot')

figure(3)
subplot(211)
stairs(t,u1),grid
ylabel('u1')
subplot(212)
stairs(t,u2),grid
ylabel('u2')

%% Report plots
figure(4)
data = load('open_loop_meas_states_N_40_alpha_02_beta_20_q_1_1.mat');
data = data.ans;

subplot(3,1,1)
plot(data(1,:), data(2,:), 'LineWidth',2)
hold on;
plot(t, x1, 'LineWidth',2);
xlabel('Time [s]')
ylabel('Travel [rad]')
title('Travel angle without feedback')
legend('Actual trajectory', 'Optimal trajectory')
grid on;

subplot(3,1,2)
plot(data(1,:), data(4,:), 'LineWidth',2)
hold on;
plot(t, x3, 'LineWidth',2);
xlabel('Time [s]')
ylabel('Pitch [rad]')
title('Pitch angle without feedback')
legend('Actual trajectory', 'Optimal trajectory')
grid on;

subplot(3,1,3)
plot(data(1,:), data(6,:), 'LineWidth',2)
hold on;
plot(t, x5, 'LineWidth',2);
xlabel('Time [s]')
ylabel('Elevation [rad]')
title('Elevation angle without feedback')
legend('Actual trajectory', 'Optimal trajectory')
grid on;

figure(5)
alpha = 0.2; beta = 20; lambda_t = 2*pi/3; lambda_k = z(1:6:N*6);
constraint = alpha * exp(-beta * (lambda_k - lambda_t).^2);
constraint = [zero_padding; constraint; zero_padding; 0];

data = load('meas_Q_10_1_1_1_40_1_q1_1_q2_01.mat');
data = data.ans;

subplot(3,1,1)
plot(data(1,:), data(2,:), 'LineWidth',2)
hold on;
plot(t, x1, 'LineWidth',2);
xlabel('Time [s]')
ylabel('Travel [rad]')
title('Travel angle with feedback')
legend('Actual trajectory', 'Optimal trajectory')
grid on;

subplot(3,1,2)
plot(data(1,:), data(4,:), 'LineWidth',2)
hold on;
plot(t, x3, 'LineWidth',2);
xlabel('Time [s]')
ylabel('Pitch [rad]')
title('Pitch angle with feedback')
legend('Actual trajectory', 'Optimal trajectory')
grid on;

subplot(3,1,3)
plot(data(1,:), data(6,:), 'LineWidth',2)
hold on;
plot(t, x5, 'LineWidth',2);
hold on;
plot(t, constraint,'--', 'LineWidth',2)
xlabel('Time [s]')
ylabel('Elevation [rad]')
title('Elevation angle with feedback')
legend('Actual trajectory', 'Optimal trajectory', 'Nonlinear constraint')
grid on;