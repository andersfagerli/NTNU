function [course_d, y_int_dot] = guidanceILOS(pos, waypoints, y_int, delta_max, gamma, kappa)
    
    x = pos(1); y = pos(2);
    
    x1 = waypoints(1,1);
    y1 = waypoints(2,1);
    x2 = waypoints(1,2);
    y2 = waypoints(2,2);
    
    y_e = crosstrackWpt(x2, y2, x1, y1, x, y);
    
    if (nargin > 3)
        delta_min = 0.1*delta_max;
        delta = (delta_max-delta_min)*exp(-gamma*(y_e)^2) + delta_min;
    else
        delta = 800;
        kappa = 1;
    end
    
    Kp = 1/delta;
    Ki = kappa*Kp;
    Pi_p = atan2(y2-y1,x2-x1);
    
    course_d = wrapTo2Pi(Pi_p - atan(Kp*y_e + Ki*y_int));
    y_int_dot = delta*y_e / (delta^2 + (y_e + kappa*y_int)^2);
end