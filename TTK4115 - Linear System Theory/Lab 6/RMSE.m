function rmse = RMSE(x_1, x_2)
    se = (x_2 - x_1).^2;
    mse = mean(se');
    rmse = sqrt(mse);
end