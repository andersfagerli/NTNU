function S = crossProdMat(n)
    if numel(n) ~= 3
        error("n must be dimension 3!")
    end

    S = [   0, -n(3), n(2);
         n(3),    0, -n(1);
        -n(2), n(1),     0];
end