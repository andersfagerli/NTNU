mutable struct DisjointSetNode
    rank::Int
    p::DisjointSetNode
    DisjointSetNode() = (obj = new(0); obj.p = obj;)
end

function findset(x::DisjointSetNode)
    if (x.p != x)
        x.p = findset(x.p)
    end
    return x.p
end

function union!(x::DisjointSetNode, y::DisjointSetNode)
    x_root = findset(x)
    y_root = findset(y)

    if (x_root.rank > y_root.rank)
        y_root.p = x_root
    else
        x_root.p = y_root
        if (x_root.rank == y_root.rank)
            y_root.rank = y_root.rank + 1
        end
    end
end
