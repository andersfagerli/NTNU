include("floyd_warshall.jl")

function create_preference_matrix(ballots,voters,candidates)
    preference_matrix = zeros(Int,candidates,candidates)
    for ballot in ballots #["ABC"]
        for candidate = 1:length(ballot) #["A" "B" "C"]
            for i = (candidate+1):length(ballot)
                preference_matrix[ballot[candidate]-'A'+1,ballot[i]-'A'+1] += 1
            end
        end
    end
    return preference_matrix
end

function find_strongest_paths(preference_matrix, candidates)
    for i = 1:candidates
        for j = 1:candidates
            if (preference_matrix[i,j] <= preference_matrix[j,i])
                preference_matrix[i,j] = 0
            else
                preference_matrix[j,i] = 0
            end
        end
    end

    strongest_paths = floyd_warshall(preference_matrix,candidates,max,min)
    return strongest_paths
end

function find_schulze_ranking(strongest_paths, candidates)
    ranking = ""
    sum_row = Array{Float64}(undef,candidates)
    for i = 1:candidates
        sum_row[i] = sum(strongest_paths[i,:])
    end
    max_indexes = reverse_index_insertion_sort(sum_row)
    for i = 1:candidates
        ranking = string(ranking,'A' + max_indexes[i]-1)
    end
    return ranking
end

function reverse_index_insertion_sort(array)
    sorted_indexes = Array{Int}(undef,length(array))
    for i = 1:length(array)
        sorted_indexes[i] = i
    end
    for i = 1:length(array)
        for j = i:length(array)
            if (array[j] > array[i])
                temp = array[i]
                array[i] = array[j]
                array[j] = temp

                sorted_indexes[i] = j
                sorted_indexes[j] = i
            end
        end
    end
    return sorted_indexes
end
