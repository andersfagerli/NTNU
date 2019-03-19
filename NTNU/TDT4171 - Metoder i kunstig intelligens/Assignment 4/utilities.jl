type Node
    attribute::Int
    children::Array{Node}
end
Node(A) = Node(A,[]) #Constructor

function Plurality(examples)
    classOne = 0
    classTwo = 0
    for class in examples[:,end]
        if (class == 1)
            classOne += 1
        else
            classTwo += 1
        end
    end

    if (classOne > classTwo)
        return Node(1)
    elseif (classOne < classTwo)
        return Node(2)
    else
        return Node(rand([1 2], 1))
    end
end

function SameClassification(examples)
    class = examples[1,end]
    for i in examples[:,end]
        if (class != i)
            return false
        end
    end
    return true
end

function Importance(attributes, examples, set_random)
    if set_random
        return rand(1:length(attributes),1)[1]
    else
        p = Positives(examples)
        n = Negatives(examples)

        max = 0
        A = 0

        for a = 1:length(attributes)
            gain = B(p/(p+n)) - Remainder(a, examples)
            if (gain > max)
                max = gain
                A = a
            end
        end

        return A
    end
end

function Remainder(A, examples)
    p = Positives(examples)
    n = Negatives(examples)
    remainder = 0

    for i = 1:2
        p_k = 0
        n_k = 0
        for j = 1:length(examples[:,A])
            if (examples[j,A] == i)
                if (examples[j,end] == 1)
                    p_k += 1
                else
                    n_k += 1
                end
            end
        end
        remainder += (p_k+n_k)/(p+n) * B(p_k/(p_k+n_k))
    end

    return remainder
end

function B(q)
    return -(q*log(2,q) + (1-q)*log(2,1-q))
end

function Positives(examples)
    p = 0
    for class in examples[:,end]
        if (class == 1.0)
            p = p + 1
        end
    end
    return p
end

function Negatives(examples)
    n = 0
    for class in examples[:,end]
        if (class == 2.0)
            n = n + 1
        end
    end
    return n
end

function ReadFile(filename)
    data = readdlm(filename)
    return data
end

function TraverseTree(A)
    for node in A.children
        list = [children.attribute for children in node.children]
        println("id: $(node.attribute), children: $list")
    end

end
