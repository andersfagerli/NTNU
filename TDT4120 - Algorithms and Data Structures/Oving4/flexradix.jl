include("counting_sort_length.jl")
include("counting_sort_letters.jl")
include("utilities.jl")

function flexradix(A, max_length)
    B = Array{String}(undef, length(A))
    A = counting_sort_length(A)

    upper = length(A)
    for i in max_length:-1:1
        while (upper > 1 && length(A[upper-1]) == i)
            upper = upper - 1
        end
        A[upper:end] = counting_sort_letters(A[upper:end],i)
    end
    return A
end
