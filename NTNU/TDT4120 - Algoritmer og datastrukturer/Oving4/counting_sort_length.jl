function counting_sort_length(A)
    B = Array{String}(undef, length(A))
    C = zeros(Int,maximum_letters(A)+1) #+1 for å ta hensyn til strenger med 0 lengde

    for i in 1:length(A)
        C[length(A[i])+1] = C[length(A[i])+1] + 1
    end

    for i in 2:length(C)
        C[i] = C[i] + C[i-1]
    end

    for i in length(A):-1:1
        B[C[length(A[i])+1]] = A[i]
        C[length(A[i])+1] = C[length(A[i])+1] - 1
    end
    return B
end

function maximum_letters(A)
    max = length(A[1])
    for i in 2:length(A)
        if (max < length(A[i]))
            max = length(A[i])
        end
    end
    return max
end
