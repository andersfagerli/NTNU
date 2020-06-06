function T = rotate_x(radians)
    c = cos(radians);
    s = sin(radians);
    T = [1, 0, 0, 0 ;
         0, c,-s, 0 ;
         0, s, c, 0 ;
         0, 0, 0, 1 ];
end