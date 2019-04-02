using Pkg
Pkg.add("DataStructures")
using DataStructures: Queue, enqueue!, dequeue!

mutable struct Node
    id::Int
    neighbours::Array{Node}
    color::Union{String, Nothing}
    distance::Union{Int, Nothing}
    predecessor::Union{Node, Nothing}
end
Node(id) = Node(id, [], nothing, nothing, nothing)


function makenodelist(adjacencylist)
    node_list = Array{Node}(undef, length(adjacencylist))
    for i = 1:length(adjacencylist)
        node_list[i] = Node(i)
    end
    for i = 1:length(adjacencylist)
        for j = 1:length(adjacencylist[i])
            push!(node_list[i].neighbours,node_list[adjacencylist[i][j]])
        end
    end
    return node_list
end

function bfs!(nodes,start)
    if (isgoalnode(start))
        return start
    end

    for i = 1:length(nodes)
        nodes[i].color = "white"
    end

    start.color = "gray"
    start.distance = 0
    queue = Queue{Node}()
    enqueue!(queue,start)

    while (length(queue) > 0)
        current = dequeue!(queue)
        for i = 1:length(current.neighbours)
            if (current.neighbours[i].color == "white")
                current.neighbours[i].color = "gray"
                current.neighbours[i].distance = current.distance + 1
                current.neighbours[i].predecessor = current
                if (isgoalnode(current.neighbours[i]))
                    return current.neighbours[i]
                end
                enqueue!(queue,current.neighbours[i])
            end
        end
        current.color = "black"
    end
    return nothing
end

function makepathto(goalnode)
    path = Int64[]
    current = goalnode
    while (current != nothing)
        prepend!(path,current.id)
        current = current.predecessor
    end
    return path
end

function printnodelist(nodelist)
    println("Skriver ut nodeliste med printnodelist")
    for node in nodelist
        neighbourlist = [neighbours.id for neighbours in node.neighbours]
        println("id: $(node.id), neighbours: $neighbourlist")
    end
end


function main(;n=5, degree=3)
    println("Kjører makenodelist")
    adjacencylist = generateadjacencylist(n, degree)
    @info "adjacencylist" adjacencylist
    nodelist = makenodelist(adjacencylist)
    printnodelist(nodelist)
end


generateadjacencylist(n, degree) = [rand(1:n, degree) for id = 1:n]


# Det kan være nyttig å kjøre mange tester for å se om programmet man har laget
# krasjer for ulike instanser
function runmanytests(;maxlistsize=15)
    # Kjører n tester for hver listestørrelse – én for hver grad
    for n = 1:maxlistsize
        for degree = 1:n
            adjacencylist = generateadjacencylist(n, degree)
            @info "Listelendge $n" n, degree #, adjacencylist
            makenodelist(adjacencylist)
        end
    end
end
