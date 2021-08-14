## A: Small boiler/large boiler - (-1/1)
## B: Without lid/with lid      - (-1/1)
## C: Heat 7/Heat 9             - (-1/1)
## Response Y - time in seconds to reach 100 degrees

library(FrF2)
plan <- FrF2(nruns=8,nfactors=3,replications=2,randomize=FALSE)

y.first =  c(235, 225, 214, 199, 233, 217, 182, 191)
y.second = c(239, 230, 208, 194, 239, 227, 185, 187)
y = c(y.first,y.second)

plan <- add.response(plan, y)

fitted <- lm(y ~ (A+B+C)^3, data=plan)
summary(fitted)

MEPlot(fitted) # main effects plot
IAPlot(fitted) # interaction effect plots
effects <- 2*fitted$coeff

#evaluating the model
library(ggplot2)
ggplot(fitted, aes(.fitted, .stdresid)) + geom_point(pch = 21) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(se = FALSE, col = "red", size = 0.5, method = "loess") +
  labs(x = "Fitted values", y = "Standardized residuals",
       title = "Fitted values vs. Standardized residuals fitted model",
       subtitle = deparse(fitted$call))

ggplot(fitted, aes(sample = .stdresid)) +
  stat_qq(pch = 19) + 
  geom_abline(intercept = 0, slope = 1, linetype = "dotted") +
  labs(x = "Theoretical quantiles", y = "Standardized residuals", 
       title = "Normal Q-Q fitted model", subtitle = deparse(fitted$call))

library(nortest) 
ad.test(rstudent(fitted))
