function T = rotate_z(radians)
    c = cos(radians);
    s = sin(radians);
    T = [c,-s, 0, 0 ;
         s, c, 0, 0 ;
         0, 0, 1, 0 ;
         0, 0, 0, 1 ];
end