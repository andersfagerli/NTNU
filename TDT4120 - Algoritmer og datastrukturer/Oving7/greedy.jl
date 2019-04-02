function can_use_greedy(coins)
    for i = 1:length(coins)-1
        if (coins[i] % coins[i+1] != 0)
            return false
        end
    end
    return true
end

function min_coins_greedy(coins,value)
    min_coins = 0
    i = 1
    while (value > 0)
        if (value - coins[i] >= 0)
            value -= coins[i]
            min_coins += 1
        else
            i += 1
        end
    end
    return min_coins
end

function min_coins_dynamic(coins,value)
    
