% Task 1c
function [u, v, theta] = extract_edges(Iu, Iv, Im, threshold)
    % Returns the u and v coordinates of pixels whose gradient
    % magnitude is greater than the threshold.

    % This is an acceptable solution for the task (you don't
    % need to do anything here). However, it results in thick
    % edges. If you want better results you can try to replace
    % this with a thinning algorithm as described in the text.
    [v,u] = find(Im > threshold);
    index = sub2ind(size(Im), v, u);
    theta = atan2(Iv(index), Iu(index));
end
