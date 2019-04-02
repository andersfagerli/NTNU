include("schulze_method.jl")

adjacency_matrix = [0 7 2; Inf 0 Inf; Inf 4 0]
nodes = 3
f = min
g = +

D = floyd_warshall(adjacency_matrix,nodes,f,g)
#T = transitive_closure(adjacency_matrix,nodes)


strongest_paths = [0 0 2; 2 0 2; 0 0 0]
ranking = find_schulze_ranking(strongest_paths,3)
println(ranking)
