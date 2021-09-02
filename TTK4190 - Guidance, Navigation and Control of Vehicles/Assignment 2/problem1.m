beta_idx = 1;
r_idx = 4;
p_idx = 3;

dutch_idxs = [beta_idx, r_idx];

A_dutch = A(dutch_idxs, dutch_idxs);

[Wn, Z, P] = damp(A_dutch);

