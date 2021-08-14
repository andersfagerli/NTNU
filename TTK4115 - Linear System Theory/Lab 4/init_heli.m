% FOR HELICOPTER NR 3-10
% This file contains the initialization for the helicopter assignment in
% the course TTK4115. Run this file before you execute QuaRC_ -> Build 
% to build the file heli_q8.mdl.

% Oppdatert høsten 2006 av Jostein Bakkeheim
% Oppdatert høsten 2008 av Arnfinn Aas Eielsen
% Oppdatert høsten 2009 av Jonathan Ronen
% Updated fall 2010, Dominik Breu
% Updated fall 2013, Mark Haring
% Updated spring 2015, Mark Haring 

%%%%%%%%%%% Physical constants
g = 9.81; % gravitational constant [m/s^2]
l_c = 0.46; % distance elevation axis to counterweight [m]
l_h = 0.66; % distance elevation axis to helicopter head [m]
l_p = 0.175; % distance pitch axis to motor [m]
m_c = 1.92; % Counterweight mass [kg]
m_p = 0.72; % Motor mass [kg]

%%%%%%%%%%% Calibration of the encoder and the hardware for the specific
%%%%%%%%%%% helicopter
Joystick_gain_x = 5;
Joystick_gain_y = -5;
V_s0 = 7.028;
K_f = (2*m_p*g*l_h - m_c*g*l_c) / (V_s0*l_h);
elevation_offset_deg = 31;

%%%%%%%%%%% Constants in equations of motion (Lab 1)
J_p = 2*m_p*l_p^2;
J_e = m_c*l_c^2 + 2*m_p*l_h^2;
J_lambda = m_c*l_c^2 + 2*m_p*(l_h^2+l_p^2);

L_1 = K_f * l_p;
L_2 = K_f * l_h;
L_3 = K_f * l_h;

K_1 = L_1 / J_p;
K_2 = L_2 / J_e;
K_3 = V_s0*L_3 / J_lambda;

%%%%%%%%%% Parameters in LQR (Lab 3)
A = [0 1 0; 0 0 0; 0 0 0];
B = [0 0; 0 K_1; K_2 0];
C = [1 0 0; 0 0 1];

Q = diag([100 1 40]);  
R = diag([1 1]);

K = lqr(A,B,Q,R);

F = inv(C*inv(B*K-A)*B);

%%%%%%%%%%% Parameters in integral effect LQR (Lab 3)
A_bar = [0 1 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 1 0 0 0 0; 0 0 1 0 0];
B_bar = [0 0; 0 K_1; K_2 0; 0 0; 0 0];
C_bar = [1 0 0 0 0; 0 0 1 0 0];

Q_bar = diag([100 1 40 1 100]);  
R_bar = diag([1 1]);

K_bar = lqr(A_bar, B_bar, Q_bar, R_bar);

F_bar = F;

step_gain = 0.3;
delay = 10;

%%%%%%%%%%% Parameters in IMU (Lab 4)
PORT = 4;
