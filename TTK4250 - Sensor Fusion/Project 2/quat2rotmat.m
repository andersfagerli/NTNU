function R = quat2rotmat(quat)  
    eta = quat(1);
    e = quat(2:4);
    S = crossProdMat(e);
    
    R = eye(3) + 2*eta*S + 2*S*S;
end