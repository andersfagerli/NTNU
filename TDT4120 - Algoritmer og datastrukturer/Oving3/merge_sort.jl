function merge_sort(A,p,r)
    if (p < r)
        q = (p+r)/2
        merge_sort(A,p,q)
        merge_sort(A,q+1,r)
        merge(A,p,q,r)
    end
end
