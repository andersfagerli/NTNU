p_idx = 3;
da_idx = 5;
a_phi1 = -A(p_idx, p_idx);
a_phi2 = A(p_idx, da_idx);

g = 9.81;
da_max = 30*pi/180; % degrees
e_phi_max = 15*pi/180; % degrees
damp_phi = 0.707;
V_g = V_a;
W = 10;
d = 1.5*pi/180;

%% Inner loop

k_p_phi = da_max / e_phi_max * sign(a_phi2);
w_n_phi = sqrt(abs(a_phi2)*da_max / e_phi_max);
k_d_phi = (2*damp_phi*w_n_phi - a_phi1) / a_phi2;
k_i_phi = -0.6;

%% Outer loop

w_n_x = w_n_phi / W;
damp_x = 0.707;
k_p_x = 2*damp_x*w_n_x*V_g/g;
k_i_x = w_n_x^2*V_g/g;
%k_i_x = 0;

fprintf("\nk_p_phi: %f\n", k_p_phi);
fprintf("k_d_phi: %f\n", k_d_phi);
fprintf("k_i_phi: %f\n", k_i_phi);
fprintf("k_p_x: %f\n", k_p_x);
fprintf("k_i_x: %f\n\n", k_i_x);

%% Root locus

s = tf('s');
den = 1;
<<<<<<< HEAD
num = -a_phi2 / ( s*(s^2 + (a_phi1 + a_phi2*k_d_phi)*s + a_phi2*k_p_phi) );
rlocus(num/den);
=======
num = a_phi2 / ( s*(s^2 + (a_phi1 + a_phi2*k_d_phi)*s + a_phi2*k_p_phi) );
rlocus(num/den, [-5:0.1:5]);

%% Problem 2f
Chi_c = 15*pi/180;
x0 = [0;0;0;0;0];
>>>>>>> 82ed3e70ff20dd7fb3fef96a99bdf068aa99eee1





