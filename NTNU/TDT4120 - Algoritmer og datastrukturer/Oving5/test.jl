include("parse_string.jl")
include("build.jl")
include("utilities.jl")

sentence = "eit en et ei ea"
parsed = parse_string(sentence)
print_array(parsed)
top_node = build(parsed)

println(max(1,2))
