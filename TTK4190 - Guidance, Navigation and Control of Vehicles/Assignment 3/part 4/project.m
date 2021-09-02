% Project in TTK4190 Guidance and Control of Vehicles 
%
% Author:           My name
% Study program:    My study program

clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USER INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h  = 0.1;    % sampling time [s]
Ns = 10000*7;  % no. of samples

psi_ref = 10 * pi/180;  % desired yaw angle (rad)
U_d = 7;                % desired cruise speed (m/s)
               
% ship parameters 
m = 17.0677e6;          % mass (kg)
Iz = 2.1732e10;         % yaw moment of inertia about CO (kg m^3)
xg = -3.7;              % CG x-ccordinate (m)
L = 161;                % length (m)
B = 21.8;               % beam (m)
T = 8.9;                % draft (m)
KT = 0.7;               % propeller coefficient (-)
Dia = 3.3;              % propeller diameter (m)
rho = 1025;             % density of water (kg/m^3)
visc = 1e-6;            % kinematic viscousity at 20 degrees (m/s^2)
eps = 0.001;            % a small number added to ensure that the denominator of Cf is well defined at u=0
k = 0.1;                % form factor giving a viscous correction
t_thr = 0.05;           % thrust deduction number

% rudder limitations
delta_max  = 40 * pi/180;        % max rudder angle      (rad)
Ddelta_max = 5  * pi/180;        % max rudder derivative (rad/s)

% added mass matrix about CO
Xudot = -8.9830e5;
Yvdot = -5.1996e6;
Yrdot =  9.3677e5;
Nvdot =  Yrdot;
Nrdot = -2.4283e10;
MA = -[ Xudot 0    0 
        0 Yvdot Yrdot
        0 Nvdot Nrdot ];

% rigid-body mass matrix
MRB = [ m 0    0 
        0 m    m*xg
        0 m*xg Iz ];
    
Minv = inv(MRB + MA); % Added mass is included to give the total inertia

% ocean current in NED
Vc = 1;                             % current speed (m/s)
betaVc = deg2rad(45);               % current direction (rad)
v_c = [Vc*cos(betaVc) Vc*sin(betaVc) 0]';

% wind expressed in NED
Vw = 10;                   % wind speed (m/s)
betaVw = deg2rad(135);     % wind direction (rad)
rho_a = 1.247;             % air density at 10 deg celsius
cy = 0.95;                 % wind coefficient in sway
cn = 0.15;                 % wind coefficient in yaw
A_Lw = 10 * L;             % projected lateral area

% linear damping matrix (only valid for zero speed)
T1 = 20; T2 = 20; T6 = 10;

Xu = -(m - Xudot) / T1;
Yv = -(m - Yvdot) / T2;
Nr = -(Iz - Nrdot)/ T6;
D = diag([-Xu -Yv -Nr]);         % zero speed linear damping

% rudder coefficients (Section 9.5)
b = 2;
AR = 8;
CB = 0.8;

lambda = b^2 / AR;
tR = 0.45 - 0.28*CB;
CN = 6.13*lambda / (lambda + 2.25);
aH = 0.75;
xH = -0.4 * L;
xR = -0.5 * L;

X_delta2 = 0.5 * (1 - tR) * rho * AR * CN;
Y_delta = 0.25 * (1 + aH) * rho * AR * CN; 
N_delta = 0.25 * (xR + aH*xH) * rho * AR * CN;   

% input matrix
Bu = @(u_r,delta) [ (1-t_thr)  -u_r^2 * X_delta2 * delta
                        0      -u_r^2 * Y_delta
                        0      -u_r^2 * N_delta            ];
                    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    
% Heading Controller
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% rudder control law
wb = 0.06;
zeta = 1;
wn = 1 / sqrt( 1 - 2*zeta^2 + sqrt( 4*zeta^4 - 4*zeta^2 + 2) ) * wb;
wref = 0.03*2;

% linearized sway-yaw model (see (7.15)-(7.19) in Fossen (2021)) used
% for controller design. The code below should be modified.
M = [m-Yvdot m*xg-Yrdot
    m*xg-Nvdot Iz-Nrdot];

N_lin = [-Yv                    (m - Xudot)*U_d
         (Xudot - Yvdot)*U_d   (m*xg - Yrdot)*U_d-Nr];

b_lin = [-2*U_d*Y_delta -2*U_d*N_delta]';

c = [0 1];

% Problem 2b), 2c)
[nom, denom] = ss2tf(-inv(M)*N_lin, inv(M)*b_lin, c, 0);
nom = nom/denom(3);
denom = denom/denom(3);

T_nomoto = 174.2045;
K_nomoto = 0.0076875;

% Problem 2d)
k = 0;
d = 1/K_nomoto;
m_control = T_nomoto/K_nomoto;

Kp = m_control * wn^2 - k;
Kd = 2*zeta*wn*m_control - d;
Ki = wn/10 * Kp;

% Propeller coefficients
num_blades = 4;         % number of propeller blades
AEAO = 0.65;            % area of blade
PD = 1.5;               % pitch/diameter ratio     
Ja = 0;                 % bollard pull
[KT, KQ] = wageningen(Ja, PD, AEAO, num_blades); %Propeller coefficients

Qm = 0;
t_T = 0.05;

% initial states
eta = [0 0 4*pi/3]'; % x y psi
nu  = [0.1 0 0]'; % u v r
eta_dot = [0 0 0]';
delta = 0;
n = 0;
xd = [0 0 0]'; % 3rd order reference model

z = 0; % integrator
y_int = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simdata = zeros(Ns+1,17);                % table of simulation data
WP = load('WP.mat');
WP = WP.WP;

waypoint_idx = 1;
waypoint_dist_threshold = 20*L;

CURRENT = true;

for i=1:Ns+1

    t = (i-1) * h;                      % time (s)
    
    % LOS guidance
    pos = eta(1:2);
    current_waypoint = WP(:, waypoint_idx);
 
    if (norm(pos-current_waypoint) < waypoint_dist_threshold)
        waypoint_idx = waypoint_idx + 1;
        if (waypoint_idx > length(WP))
            waypoint_idx = waypoint_idx - 1;
        end
    end
    
    waypoints = WP(:, (waypoint_idx-1):waypoint_idx);
    
    %course_d = guidance(pos, waypoints, U_d/0.001, 10);
    [course_d, y_int_dot] = guidanceILOS(pos, waypoints, y_int, U_d/0.001, 10, 1); 
    
    crab_angle = atan2(nu(2),nu(1)); % for crab angle compensation
    psi_ref = course_d; % - crab_angle;
    
    % 3rd order reference model
    Ad = [0     1                   0;
          0     0                   1;
          -wref^3 -(2*zeta+1)*wref^2 -(2*zeta+1)*wref];
      
    Bd = [0; 0; wref^3];
    
    xd_dot = Ad * xd + Bd * psi_ref;
    
    
    R = Rzyx(0,0,eta(3));
    
    % current (should be added here)
    v_c = [Vc*cos(betaVc - eta(3)) Vc*sin(betaVc - eta(3)) 0];
    nu_c = v_c';
    u_c = Vc*cos(betaVc - eta(3));
    
    if CURRENT
        nu_r = nu - nu_c;
    else
        nu_r = nu;
    end
    
    % wind (should be added here)
    if t > 200
        u_rw = nu(1) - Vw*cos(betaVw);
        v_rw = nu(2) - Vw*sin(betaVw);
        
        V_rw = sqrt(u_rw^2 + v_rw^2);
        gamma_rw = -atan2(v_rw, u_rw);
        
        q = 0.5*rho_a*V_rw^2;
        
        Ywind = q*cy*sin(gamma_rw)*A_Lw;
        Nwind = q*cn*sin(2*gamma_rw)*A_Lw*L;
    else
        Ywind = 0;
        Nwind = 0;
    end
    tau_env = [0 Ywind Nwind]';
    
    % state-dependent time-varying matrices
    CRB = m * nu(3) * [ 0 -1 -xg 
                        1  0  0 
                        xg 0  0  ];
                    
    % coriolis due to added mass
    CA = [  0   0   Yvdot * nu_r(2) + Yrdot * nu_r(3)
            0   0   -Xudot * nu_r(1) 
          -Yvdot * nu_r(2) - Yrdot * nu_r(3)    Xudot * nu_r(1)   0];
    N = CRB + CA + D;
    
    % nonlinear surge damping
    Rn = L/visc * abs(nu_r(1));
    Cf = 0.075 / ( (log(Rn) - 2)^2 + eps);
    Xns = -0.5 * rho * (B*L) * (1 + k) * Cf * abs(nu_r(1)) * nu_r(1);
    
    % cross-flow drag
    Ycf = 0;
    Ncf = 0;
    dx = L/10;
    Cd_2D = Hoerner(B,T);
    for xL = -L/2:dx:L/2
        vr = nu_r(2);
        r = nu_r(3);
        Ucf = abs(vr + xL * r) * (vr + xL * r);
        Ycf = Ycf - 0.5 * rho * T * Cd_2D * Ucf * dx;
        Ncf = Ncf - 0.5 * rho * T * Cd_2D * xL * Ucf * dx;
    end
    d = -[Xns Ycf Ncf]';
    
    psi_d = xd(1);
    r_d = xd(2);
    u_d = U_d;
   
    
    % thrust 
    thr = rho * Dia^4 * KT * abs(n) * n;    % thrust command (N)
        
    % control law
    psi_e = eta(3) - psi_d;
    psi_dot_e = nu(3) - r_d;
    z_dot = psi_e;
    
    delta_c = - (Kp*ssa(psi_e) + Kd*psi_dot_e + Ki*ssa(z));
    
    % tau_ff = m*1/T_nomoto *r_d;
    % tau_N = -tau_env(3) + tau_ff - (Kp*ssa(psi_e) + Kd*psi_dot_e + Ki*ssa(z));
    % delta_c = tau_N / N_delta; % rudder angle command (rad), see (15.162)
    
    % ship dynamics
    u = [ thr delta ]';
    tau = Bu(nu_r(1),delta) * u;
    nu_dot = Minv * (tau_env + tau - N * nu_r - d); 
    eta_dot = R * nu;    
    
    % Rudder saturation and dynamics (Sections 9.5.2)
    if abs(delta_c) >= delta_max
        delta_c = sign(delta_c)*delta_max;
    end
    
    delta_dot = delta_c - delta;
    if abs(delta_dot) >= Ddelta_max
        delta_dot = sign(delta_dot)*Ddelta_max;
    end    
    
    % propeller dynamics
    Im = 100000; Tm = 10; Km = 0.6;         % propulsion parameters
    n_c = 10;                               % propeller speed (rps)
    T_prop = rho * Dia^4 * KT * abs(n) * n;
    Q_prop = rho * Dia^5 * KQ * abs(n) * n;
    T_d = (U_d - u_c)*Xu / (t_T - 1);
    n_d = sign(T_d) * sqrt( abs(T_d) / (rho*Dia^4*KT) );
    Q_d = rho * Dia^5 * KQ * abs(n_d) * n_d;
    Y = (1 / Km) * Q_d;
    Qm_dot = (1 / Tm) * (-Qm + Y*Km);
    Qf = 0;
    
    n_dot = (1/Im) * (Qm - Q_prop - Qf); 
    
    % store simulation data in a table (for testing)
    side_slip = asin(nu_r(2)/norm(nu_r));
    
    simdata(i,:) = [t n_c delta_c n delta eta' nu' u_d psi_d r_d side_slip crab_angle course_d];       
     
    % Euler integration
    eta = euler2(eta_dot,eta,h);
    nu  = euler2(nu_dot,nu,h);
    delta = euler2(delta_dot,delta,h);   
    n  = euler2(n_dot,n,h);
    xd = euler2(xd_dot,xd,h);
    z = euler2(z_dot,z,h);
    Qm = euler2(Qm_dot,Qm,h);
    y_int = euler2(y_int_dot,y_int,h);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t       = simdata(:,1);                 % s
n_c     = 60 * simdata(:,2);            % rpm
delta_c = (180/pi) * simdata(:,3);      % deg
n       = 60 * simdata(:,4);            % rpm
delta   = (180/pi) * simdata(:,5);      % deg
x       = simdata(:,6);                 % m
y       = simdata(:,7);                 % m
psi     = (180/pi) * simdata(:,8);      % deg
u       = simdata(:,9);                 % m/s
v       = simdata(:,10);                % m/s
r       = (180/pi) * simdata(:,11);     % deg/s
u_d     = simdata(:,12);                % m/s
psi_d   = (180/pi) * simdata(:,13);     % deg
r_d     = (180/pi) * simdata(:,14);     % deg/s
side_slip = (180/pi) * simdata(:,15);   % deg
crab_angle = (180/pi) * simdata(:,16);  % deg
course_d = (180/pi) * simdata(:,17);    % deg
course  = (psi+crab_angle);             % deg

figure(1)
figure(gcf)
subplot(311)
plot(y,x,'linewidth',2); axis('equal')
title('North-East positions (m)'); xlabel('time (s)'); 
subplot(312)
plot(t,psi,t,psi_d,'linewidth',2);
title('Actual and desired yaw angles (deg)'); xlabel('time (s)');
subplot(313)
plot(t,r,t,r_d,'linewidth',2);
title('Actual and desired yaw rates (deg/s)'); xlabel('time (s)');

figure(2)
figure(gcf)
subplot(311)
plot(t,u,t,u_d,'linewidth',2);
title('Actual and desired surge velocities (m/s)'); xlabel('time (s)');
subplot(312)
plot(t,n,t,n_c,'linewidth',2);
title('Actual and commanded propeller speed (rpm)'); xlabel('time (s)');
subplot(313)
plot(t,delta,t,delta_c,'linewidth',2);
title('Actual and commanded rudder angles (deg)'); xlabel('time (s)');

figure(3) 
figure(gcf)
subplot(211)
plot(t,u,'linewidth',2);
title('Actual surge velocity (m/s)'); xlabel('time (s)');
subplot(212)
plot(t,v,'linewidth',2);
title('Actual sway velocity (m/s)'); xlabel('time (s)');

figure(4)
hold on
siz=size(WP);
for ii=1:(siz(2)-1)   
    plot([WP(2,ii), WP(2,ii+1)], [WP(1,ii), WP(1,ii+1)], 'r-x')
end
plot(y,x,'linewidth',2); axis('equal')
title('North-East positions (m)');

figure(5)
plot(t, course, t, course_d, t, psi, 'linewidth', 2)
legend('\chi (deg)', '\chi_d (deg)', '\psi (deg)')
xlabel('time (s)');
ylabel('Angle');
title('Course, desired course and heading')

figure(6)
plot(t, side_slip, t, crab_angle, 'linewidth', 2)
legend('\beta (deg)', '\beta_c (deg)')
xlabel('time (s)');
ylabel('Angle');
title('Side slip vs. crab angle')




