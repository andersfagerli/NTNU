include ("disjoint_set.jl")

function findclusters(E::Vector{Tuple{Int, Int, Int}}, n::Int, k::Int)
    #Tuple{w,u,v}
    A = Array{Tuple{Int,Int}}(undef,0)
    clusters = Array{Array{Int}}(undef,k)
    sort!(E)
    nodes = Array{DisjointSetNode}(undef,n)
    for i = 1:n
        nodes[i] = DisjointSetNode()
    end

    for edge in E
        if (findset(edge(2)) != findset(edge(3)))
            push!(A,(edge(2),edge(3)))
            union!(nodes[edge(2)],nodes[edge(3)])
        end
        if (length(A) == (n-1-k+1))
            break
        end
    end

    
