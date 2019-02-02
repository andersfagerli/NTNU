struct Node
    children::Dict{Char,Node}
    posi::Array{Int}
end

function build(list_of_words)
    root = Node(Dict(),[])
    for i = 1:length(list_of_words)
        parent = root
        character = list_of_words[i][1][1]
        if (!haskey(parent.children, character)) #Make new child node if key(character) doesn't exist in dict yet
            parent.children[character] = Node(Dict(),[])
        end
        if (length(list_of_words[i][1]) > 1)
            for j = 2:length(list_of_words[i][1])
                parent = parent.children[character]
                character = list_of_words[i][1][j]
                if (!haskey(parent.children, character))
                    parent.children[character] = Node(Dict(),[])
                end
            end
        end
        push!(parent.children[character].posi, list_of_words[i][2])
    end
    return root
end

function positions(word,node,index=1)
    if (length(word) > 0 && haskey(node.children,word[index]))
        if (length(word) > 0 && word[index] == '?')
            for i = 'a':'z'
                #word[index] = i
                word = string(i,word[(index+1):end])
                return positions(word,node)
            end
        else
            return positions(word[(index+1):length(word)], node.children[word[index]])
        end
    elseif (length(word) > 0 && !haskey(node.children,word[index]))
        return Int64[] #Ordet finnes ikke i treet
    else
        return node.posi
    end
end
