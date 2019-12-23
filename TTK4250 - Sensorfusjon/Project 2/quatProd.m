function qprod = quatProd(ql, qr)

    if numel(ql) == 3 % assume pure quat
        ql = [0; ql];
    end
    
    if numel(qr) == 3 % assume pure quat
        qr = [0; qr];
    end
    
    eta_l = ql(1);
    e_l = ql(2:4);
    
    eta_r = qr(1);
    e_r = qr(2:4);
    
    qprod = [eta_l*eta_r - e_l'*e_r ; 
             eta_r*e_l + eta_l*e_r + crossProdMat(e_l)*e_r];
end

