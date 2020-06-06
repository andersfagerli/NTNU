function [row,col] = extract_peaks(H, window_size, threshold)
    D = imdilate(H, ones(window_size));
    P = (H >= D) & (H >= threshold);
    [row,col,~] = find(P);
end