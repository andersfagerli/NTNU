include("counting_sort_letters.jl")
include("counting_sort_length.jl")
include("flexradix.jl")
include("utilities.jl")

A = ["kobra", "aggie", "agg", "kort", "hyblen"]
println(flexradix(A,6))
