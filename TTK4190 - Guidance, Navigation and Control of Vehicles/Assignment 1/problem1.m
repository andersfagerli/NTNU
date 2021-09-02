%% Constants

m = 180; % kg
R33 = 2.0; % m

%% 1.1

% x = [e' w']';
% e = x(1:3);
% w = x(4:6);

% Skew-symmetric matrix
S = @(v) [    0, -v(3),  v(2);
           v(3),     0, -v(1);
          -v(2),  v(1),     0];

% Scalar part of unit quaternion as function of vector part 
eta = @(e) sqrt(1 - e'*e);

% Linearized A
A = @(x) [-0.5*S(x(4:6)), 0.5*(eta(x(4:6))*eye(3) + S(x(4:6)));
             zeros(3, 6)];

% Linearized B
B = [zeros(3);
     1/(m*R33^2)*eye(3)];         

% Equilibrium x
x = zeros(6,1);

%% 1.2

kp = 2;
kd = 40;

K = [-kp*eye(3), -kd*eye(3)];

A_closed = A(x) + B*K;
[V, D] = eig(A_closed);
x = real(diag(D))
y = imag(diag(D))
plot(x, y, 'x')
grid on