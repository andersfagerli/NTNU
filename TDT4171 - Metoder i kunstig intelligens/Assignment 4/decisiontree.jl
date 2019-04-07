############################
### Using Julia v. 0.4.5 ###
############################

### Notes: Only works on binary attributes (1,2) and classes

include("utilities.jl")

function DecisionTreeLearning(examples, attributes, parent_examples, set_random)
    if isempty(examples)
        return Plurality(parent_examples)
    elseif SameClassification(examples)
        return Node(examples[1,end])
    elseif isempty(attributes)
        return Plurality(examples)
    else
        A = Importance(attributes,examples,set_random)
        tree = Node(A)
        splice!(attributes, find(x->x==A,attributes)[1]) #Remove element A
        for v_k = 1:2
            exs = Matrix(0,length(examples[1,:]))
            for i = 1:length(examples[:,end])
                if (examples[i,A] == v_k)
                    exs = [exs; examples[i,:]] #Appending a row to a matrix
                end
            end
            subtree = DecisionTreeLearning(exs, attributes, examples, set_random)
            push!(tree.children, subtree)
        end
        return tree
    end
end

function main()
    training_examples = ReadFile("training.txt")
    test_examples = ReadFile("test.txt")
    attributes = [1,2,3,4,5,6,7]

    randTree = DecisionTreeLearning(training_examples,attributes, [], true)
    printout2 = TraverseTree(randTree)

    attributes = [1,2,3,4,5,6,7]
    tree = DecisionTreeLearning(training_examples, attributes, [], false)
    printout1 = TraverseTree(tree)

    println(TestClassifier(tree, test_examples))
    println(TestClassifier(randTree, test_examples))

    println(printout1)
    println(printout2)

end

main()
