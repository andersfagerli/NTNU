function find_median(A,lower,upper)
    lower_index = bisect_left(A,1,length(A)+1,lower)
    upper_index = bisect_right(A,1,length(A)+1,upper)-1

    if ((upper_index - lower_index + 1) % 2 == 0)
        index1 = floor(Int,(upper_index+lower_index)/2)
        index2 = floor(Int,(upper_index+lower_index)/2)+1
        return (A[index1]+A[index2])/2
    else
        return A[ceil(Int,(upper_index+lower_index)/2)]
    end
end

function bisect_left(A, p, r, v)
    i = p
    if p < r
       q = floor(Int, (p+r)/2)  # floor() er en innebygd funksjon som runder ned. ceil() runder opp.
       if v <= A[q]
           i = bisect_left(A, p, q, v)
       else
           i = bisect_left(A, q+1, r, v)
       end
    end
    return i
end

function bisect_right(A,p,r,v)
    i = p
    if (p < r)
        q = floor(Int, (p+r)/2)
        if (v >= A[q])
            i = bisect_right(A,q+1,r,v)
        else
            i = bisect_right(A,p,q,v)
        end
    end
    return i
end
