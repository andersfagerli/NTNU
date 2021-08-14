include("utilities.jl")

function parse_string(sentence)
    parsed = split(sentence, " ");
    parsed_string = Array{Any}(undef,length(parsed))
    prev_length = 1
    for i = 1:length(parsed_string)
        parsed_string[i] = (parsed[i],prev_length + i - 1)
        prev_length += length(parsed[i])
    end
    return parsed_string
end
