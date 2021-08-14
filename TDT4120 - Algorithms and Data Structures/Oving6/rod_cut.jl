function maximum_profit(N,prices)
    if N == 0
        return 0
    end
    q = -1
    for i = 1:N
        q = max(q, prices[i] + maximum_profit(N-i,prices))
    end
    return q
end
