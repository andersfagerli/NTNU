function [R,t] = choose_solution(uv1, uv2, K1, K2, Rts)
    % Chooses among the rotation and translation solutions Rts
    % the one which gives the most points in front of both cameras.

    % todo: Choose the correct solution
    soln = 1;
    fprintf('Choosing solution %d.\n', soln);
    R = Rts(1:3, 1:3, soln);
    t = Rts(1:3, 4, soln);
end
