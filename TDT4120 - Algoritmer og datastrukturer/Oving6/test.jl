include("rod_cut.jl")
include("seam_carving.jl")
include("utilities.jl")

weights = [3  6  8 6 3;
           7  6  5 7 3;
           4  10 4 1 6;
           10 4  3 1 2;
           6  1  7 3 9]

path_weights = cumulative(weights)
print_matrix(path_weights)
path = back_track(path_weights)
print_array(path)
