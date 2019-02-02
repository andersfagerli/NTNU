include("utilities.jl")

include("algdat_sort.jl")
include("find_median.jl")


#=A = zeros(Int8, 1, 20)
for i in 1:length(A)
    A[i] = rand(1:50)
end
print_array(A)

#partition!(A,1,length(A))
#print_array(A)

quicksort!(A,1,length(A))
print_array(A)=#
println(find_median([1, 5, 5, 7, 10, 10, 10], 5, 10))
