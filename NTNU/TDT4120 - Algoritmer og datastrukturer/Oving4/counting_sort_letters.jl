function counting_sort_letters(A, position)
    B = Array{String}(undef, length(A))
    C = zeros(26)

    for i in 1:length(A)
        C[A[i][position]-'a'+1] = C[A[i][position]-'a'+1] + 1
    end

    for i in 2:length(C)
        C[i] = C[i] + C[i-1]
    end

    for i in length(A):-1:1
        B[convert(Int,C[A[i][position]-'a'+1])] = A[i]
        C[A[i][position]-'a'+1] = C[A[i][position]-'a'+1] - 1
    end
    return B
end
