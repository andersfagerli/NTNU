function draw_line(theta, rho)
    ulim = xlim();
    vlim = ylim();
    c = cos(theta);
    s = sin(theta);
    if abs(s) > abs(c)
        u1 = ulim(1);
        u2 = ulim(2);
        v1 = (rho-u1*c)/s;
        v2 = (rho-u2*c)/s;
        [v1,u1] = clamp_line_parameters(v1, u1, vlim(1), vlim(2), rho, s, c);
        [v2,u2] = clamp_line_parameters(v2, u2, vlim(1), vlim(2), rho, s, c);
    else
        v1 = vlim(1);
        v2 = vlim(2);
        u1 = (rho-v1*s)/c;
        u2 = (rho-v2*s)/c;
        [u1,v1] = clamp_line_parameters(u1, v1, ulim(1), ulim(2), rho, c, s);
        [u2,v2] = clamp_line_parameters(u2, v2, ulim(1), ulim(2), rho, c, s);
    end
    plot([u1, u2], [v1, v2], 'yellow');
end

function [a, b] = clamp_line_parameters(a, b, a_min, a_max, rho, A, B)
    if a < a_min || a > a_max
        a = max(a_min, min(a_max, a));
        b = (rho - a*A)/B;
    end
end
