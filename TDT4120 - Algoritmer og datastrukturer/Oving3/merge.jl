function merge(A,p,q,r)
    n_1 = q-p+1
    n_2 = r-q

    L = Array{Int64}(undef, n_1)
    R = Array{Int64}(undef, n_2)

    for i in 1:(n_1)
        L[i] = A[i]
        R[i] = A[n_1+i]
    end
    

end
