function [xmix, Pmix] = reduceGaussMix(w, x, P)
    w = w(:);
    M = numel(w);
    n = size(x, 1);
    
    % allocate
    xmix = zeros(n, 1);
    Pmix = zeros(n, n);
    
    % mean
    for i = 1:M
        xmix = xmix + x(:, i) * w(i);
    end
    % covariance
    for i = 1:M
        mixInnovation = x(:, i) - xmix;
        Pmix = Pmix + (P(:, :, i) + mixInnovation * mixInnovation') * w(i);
    end
end
