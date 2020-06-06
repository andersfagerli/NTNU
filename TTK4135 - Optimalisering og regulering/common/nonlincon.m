function [ c, ceq ] = nonlincon( z )
    N = 40;
    lambda_k = z(1:6:N*6);
    e_k = z(5:6:N*6);
    alpha = 0.2;
    beta = 20;
    lambda_t = 2*pi/3;

    c = alpha * exp(-beta * (lambda_k - lambda_t).^2) - e_k;
    ceq = [];
end

