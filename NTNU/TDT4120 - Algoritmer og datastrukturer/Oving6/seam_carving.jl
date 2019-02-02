function cumulative(weights)
    rows,cols = size(weights)
    for i = 2:rows
        for j = 1:cols
            if (j == 1)
                weights[i,j] += min(weights[i-1,j],weights[i-1,j+1])
            elseif (j == cols)
                weights[i,j] += min(weights[i-1,j],weights[i-1,j-1])
            else
                weights[i,j] += min(weights[i-1,j],weights[i-1,j+1],weights[i-1,j-1])
            end
        end
    end
    return weights
end

function back_track(path_weights)
    rows,cols = size(path_weights)
    tuples = Array{Any}(undef,rows)

    min_weight = Inf
    min_index = Inf
    for i = cols:-1:1
        if (path_weights[rows,i] <= min_weight)
            min_weight = path_weights[rows,i]
            min_index = i
        end
    end

    tuples[1] = (rows,min_index)
    for j = rows-1:-1:1
        min_weight = Inf
        temp_index = Inf
        right = 1
        left = 1
        if (min_index == 1)
            left = 0
        elseif (min_index == cols)
            right = 0
        end

        for i = (min_index+right):-1:(min_index-left)
            if (path_weights[j,i] <= min_weight)
                min_weight = path_weights[j,i]
                temp_index = i
            end
        end

        min_index = temp_index
        tuples[rows-j+1] = (j,min_index)
    end
    return tuples
end
