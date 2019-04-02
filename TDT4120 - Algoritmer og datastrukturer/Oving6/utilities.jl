function print_matrix(A)
    rows,cols = size(A)
    for i = 1:rows
        for j = 1:cols
            print(A[i,j])
            print(" ")
        end
        println("")
    end
end

function print_array(A)
    for i = 1:length(A)
        print(A[i])
        print(" ")
    end
    println("")
end
