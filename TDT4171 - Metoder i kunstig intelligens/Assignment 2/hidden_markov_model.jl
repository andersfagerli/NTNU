#Written in Julia, a language that shares syntax between Python and Matlab
#List indices start at 1, matrix multiplication naturally supported


function normalize(vec)
    alpha = 1/sum(vec)
    return alpha*vec
end

function forward(f,O,T)
    #Equation 15.12
    prob = O*T*f
    return normalize(prob)
end

function backward(b,O,T)
    #Equation 15.13
    prob = T*O*b
    return prob
end

#Similar to psuedo-code given in Figure 15.4
function forward_backward(ev,prior,O,T)
    #Initializing array of vectors. Typename "Any" means the arrays may consist of any type of variable
    fv = Array{Any}(1,length(ev)+1) # 1x(length(ev)+1) array
    sv = Array{Any}(1,length(ev))   # 1xlength(ev) array

    #Initial backward and forward messages
    b = [1,1]
    fv[1] = prior

    for i = 2:length(ev)+1
        e = ev[i-1]+1;  #Compensate for indexing starting at 1
        fv[i] = forward(fv[i-1],O[e],T)
    end

    for i = length(ev):-1:1
        println("b_",i,": ",b) #Documenting backward messages
        e = ev[i]+1
        sv[i] = normalize(fv[i+1].*b) #Pointwise multiplication by .*
        b = backward(b,O[e],T)
    end

    return sv
end

function main()
    #Dynamic model
    T = [0.7 0.3; 0.3 0.7]
    #Observation model, written as two matrices
    O = Array[[0.1 0.0; 0.0 0.8],[0.9 0.0;0.0 0.2]]

    ### Part B
    println("Part B")

    #Initial probabilities of rain/rain'
    f = [0.5;0.5]
    #Sequence of observations
    ev = [1 1 0 1 1]
    for i = 1:length(ev)
        e = ev[i]+1
        f = forward(f,O[e],T)
        println("Day ", i, ": ", f)
    end

    ### Part C
    println("Part C")

    f = [0.5;0.5]
    ev = [1 1]
    sv = forward_backward(ev,f,O,T)
    println("Day 1: " ,sv[1])

    ev = [1 1 0 1 1]
    sv = forward_backward(ev,f,O,T)
    for i = 1:length(sv)
        println("Day ",i,": ",sv[i])
    end
end

#Run main()
main()
