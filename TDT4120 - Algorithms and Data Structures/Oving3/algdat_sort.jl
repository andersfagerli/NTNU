function algdat_sort!(A)
    p = 1
    r = length(A)
    quicksort!(A,p,r)
end

function quicksort!(A,p,r)
    if p < r
        q = partition!(A,p,r)
        quicksort!(A,p,q-1)
        quicksort!(A,q+1,r)
    end
end

function partition!(A,p,r)
    random_index = rand(p:r)
    pivot = A[random_index]
    A[random_index] = A[r]
    A[r] = pivot

    j = p-1
    for i in p:r-1
        if (A[i] <= pivot)
            j = j+1
            exchange!(A,i,j)
        end
    end
    exchange!(A,j+1,r)

    return j+1
end

function exchange!(A,index_1,index_2)
    temp = A[index_1]
    A[index_1] = A[index_2]
    A[index_2] = temp
end
