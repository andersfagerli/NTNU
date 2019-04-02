function floyd_warshall(adjacency_matrix, nodes, f, g)
    n = nodes
    D = Array{Array{Any,2}}(undef,0)
    push!(D,adjacency_matrix)
    for k = 2:n+1
        d_k = Array{Any,2}(undef,n,n)
        for i = 1:n
            for j = 1:n
                d_k[i,j] = f(D[k-1][i,j], g(D[k-1][i,k-1], D[k-1][k-1,j]))
            end
        end
        push!(D, d_k)
    end
    return D[length(D)]
end

function transitive_closure(adjacency_matrix,nodes)
    return floyd_warshall(adjacency_matrix,nodes,|,&)
end
