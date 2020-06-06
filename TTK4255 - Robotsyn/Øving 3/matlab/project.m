function uv = project(K, X)
    uvw = K*X(1:3,:);
    wu = uvw(1,:);
    wv = uvw(2,:);
    w  = uvw(3,:);
    uv = [wu./w ; wv./w];
end
