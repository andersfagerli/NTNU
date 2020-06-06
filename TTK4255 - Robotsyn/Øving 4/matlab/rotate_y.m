function T = rotate_y(radians)
    c = cos(radians);
    s = sin(radians);
    T = [ c, 0, s, 0 ;
          0, 1, 0, 0 ;
         -s, 0, c, 0 ;
          0, 0, 0, 1 ];
end