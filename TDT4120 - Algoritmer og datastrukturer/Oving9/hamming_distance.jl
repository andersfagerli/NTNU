function hammingdistance(s1::String,s2::String)
    distance = 0
    for i = 1:length(s1)
        if (s1[i] != s2[i])
            distance += 1
        end
    end
    return distance
end
