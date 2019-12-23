#################
### Problem 1 ###
#################

### Problem 1c)
## c.1)

# Time period
days = 365; # Ignoring leap years
years = 5;

# Rates
lambda = 1/100;
mu = 1/7;
rates = c(lambda, mu);

# State (1: Susceptible, 2: Infected)
xVals = numeric(days*years)
xVals[1] = 1; # Start as susceptible
i = 1;
tTimes = c(0);
totTime = 0;

while (totTime <= (days*years)) {
    currState = xVals[i];
    sjTime = rexp(1, rate = rates[currState]);
    nextState = currState %% 2 + 1; # Switch state

    xVals[i+1] = nextState;
    tTimes = c(tTimes, tail(tTimes,1)+sjTime);
    
    totTime = totTime + sjTime;
    i = i + 1;
}

# Plotting

pdf('problem1c.pdf')
par(mar=c(5,6,4,1)+.1)
plot(NULL, NULL, xlim = c(0, days*years), ylim = c(0.8, 2.2), xlab = "Time (days)", ylab = "State", cex.lab = 1.5, cex.axis = 1.5)
  for(i in 1:(length(xVals)-1)){
    lines(tTimes[i:(i+1)], rep(xVals[i], 2), lwd = 4, type = "l")
  }
  lines(tail(tTimes, 1) + c(0,1), c(1,1), lwd = 4)



## c.2)

# Time period
years = 1000;

# State
xVals = numeric(days*years);
xVals[1] = 1;
i = 1;
totTime = 0;      # Total time the simulation runs
infectedTime = 0; # Time the individual is infected out of the total time

while (totTime <= (days*years)) {
  currState = xVals[i];
  sjTime = rexp(1, rate = rates[currState]);
  nextState = currState %% 2 + 1; # Switch state

  xVals[i+1] = nextState;

  if (currState == 2) { # If infected; update infected time
    infectedTime = infectedTime + sjTime;
  }
  
  totTime = totTime + sjTime;
  i = i + 1;
}

fracInfected = infectedTime / totTime;
print(fracInfected)

################################
### Problem 2 #################
###############################

# Functions and variables for later computations

library(Matrix)

create.Covar.Matrix <- function(theta, phi.M, sigma) {
    ones <- as.matrix(rep(1.0, length(theta)));
    H <- abs(theta %*% t(ones) - ones %*% t(theta));
    return(sigma*(1 + phi.M*H)*exp(-phi.M*H));
}

create.Mu.C <- function(mu.1, mu.2, a, covar.matrix) {
    p <- length(mu.1);
    q <- length(mu.2);
    N <- p + q;
    covar.matrix.11 <- covar.matrix[1:p, 1:p];
    covar.matrix.12 <- covar.matrix[1:p, (p+1):N];
    covar.matrix.21 <- covar.matrix[(p+1):N, 1:p];
    covar.matrix.22 <- covar.matrix[(p+1):N, (p+1):N];
    return(mu.1 + (covar.matrix.12 %*% solve(covar.matrix.22)) %*% (a - mu.2))
}

create.Covar.Matrix.Conditional <- function(p, q, covar.matrix) {
    N <- p + q;
    covar.matrix.11 <- covar.matrix[1:p, 1:p];
    covar.matrix.12 <- covar.matrix[1:p, (p+1):N];
    covar.matrix.21 <- covar.matrix[(p+1):N, 1:p];
    covar.matrix.22 <- covar.matrix[(p+1):N, (p+1):N];
    return(covar.matrix.11 - covar.matrix.12 %*% (solve(covar.matrix.22) %*% covar.matrix.21))
}

get.pred.interval <- function(mean, var, alpha) {
    zq <- qnorm(alpha/2, lower.tail=FALSE);
    lower <- mean - zq*sqrt(var);
    upper <- mean + zq*sqrt(var);
    return(cbind(lower, upper));
}



#############
# Problem a
#############


phi.M <- 15;
E.Y <- 0.5;
theta.cond <- as.matrix(c(0.300, 0.350, 0.390, 0.410, 0.450));
theta.grid <- as.matrix(seq(from=0.250, to=0.500, by=0.005));
theta <- rbind(theta.grid, theta.cond)

# Useful constants for lengths

l.tg <- length(theta.grid);
l.tc <- length(theta.cond);
N <- l.tg + l.tc;

mu <- as.matrix(rep(E.Y, N))
y.cond <- as.matrix(c(0.500, 0.320, 0.400, 0.350, 0.600));
sigma <- 0.5^2;
covar.mat <- create.Covar.Matrix(theta, phi.M, sigma);

mu.uncond <- mu[1:l.tg];
mu.cond.on <- mu[(l.tg + 1):N];
mu.cond <- create.Mu.C(mu.uncond, mu.cond.on, y.cond, covar.mat);

covar.mat.cond <- create.Covar.Matrix.Conditional(l.tg, l.tc, covar.mat);

var <- diag(covar.mat.cond);
# Correct for floating point errors, som diagonal elements are ~ -1e-16
var[var < 0] <- 0;

task <- 'task2a.pdf'

conf.interval <- 90; # % 
alpha <- 1 - conf.interval / 100;
range <- get.pred.interval(mu.cond, var, alpha);
lower <- range[,1];
upper <- range[,2];

pdf(task);
par(mar=c(5,6,4,1)+.1)
plot(NULL,NULL, xlim = c(0.25,0.5), ylim = c(0.2, 1.0), main = "Expected value with 90% confidence interval",
     xlab = expression(paste(theta)), ylab = expression(paste(E(Y(theta)))), cex.lab = 1.5)
lines(theta.grid, mu.cond, col="black", lwd=1.2*2)
lines(theta.grid,upper,lty=2,col="blue", lwd=1.2*2)
lines(theta.grid,lower,lty=2,col="green", lwd=1.2*2)
points(theta.cond,y.cond,col = "red", pch = 19)
legend(0.35,0.9,legend = c(expression(paste(mu[cond])),"upper prediction bound", "lower prediction bound", "Values conditioned on"),
       col = c("black","blue","green", "red"), cex = 0.8, lty = c(1,2,2,NA), lwd=2.4, pch=c(NA, NA, NA, 19))
dev.off();


################
### Problem b
################

task <- 'task2b.pdf'

threshold <- 0.30;
probs <- pnorm(threshold, mean=mu.cond, sd=sqrt(var));

pdf(task)
par(mar=c(5,6,4,1)+.1)
plot(NULL,NULL, xlim = c(0.25,0.5), ylim = c(0, 0.4), main = expression(paste(P(Y(theta)<0.30))),
     xlab = expression(paste(theta)), ylab = expression(paste(P(Y(theta)))), cex.lab = 1.5)
lines(theta.grid, probs, col="blue", lwd=1.2*2)
legend(0.35,0.9,legend = expression(paste(P(Y(theta)<0.30))), cex = 0.8, lty = 1, lwd=2.4, col="blue")
dev.off();


##########################
### Problem c
##########################


# Recalculate new conditional mean and variance
theta.cond.updated <- as.matrix(c(as.vector(theta.cond), 0.33));
y.cond.updated <- as.matrix(c(as.vector(y.cond), 0.40));
theta <- rbind(theta.grid, theta.cond.updated)

# Useful constants for lengths

l.tg <- length(theta.grid);
l.tc <- length(theta.cond.updated);
N <- l.tg + l.tc;

mu <- as.matrix(rep(E.Y, N))
sigma <- 0.5^2;
covar.mat.updated <- create.Covar.Matrix(theta, phi.M, sigma);

mu.uncond <- mu[1:l.tg];
mu.cond.on <- mu[(l.tg + 1):N];
mu.cond.updated <- create.Mu.C(mu.uncond, mu.cond.on, y.cond.updated, covar.mat.updated);

covar.mat.cond.updated <- create.Covar.Matrix.Conditional(l.tg, l.tc, covar.mat.updated);

var.updated <- diag(covar.mat.cond.updated);
# Correct for floating point errors, som diagonal elements are ~ -1e-16
var.updated[var.updated < 0] <- 0;

conf.interval <- 90; # % 
alpha <- 1 - conf.interval / 100;
range <- get.pred.interval(mu.cond.updated, var.updated, alpha);
lower <- range[,1];
upper <- range[,2];

task <- 'task2c_pred.pdf'

pdf(task)
par(mar=c(5,6,4,1)+.1)
plot(NULL,NULL, xlim = c(0.25,0.5), ylim = c(0.2, 1.0), main = "Expected value with 90% confidence interval",
     xlab = expression(paste(theta)), ylab = expression(paste(E(Y(theta)))), cex.lab = 1.5)
lines(theta.grid, mu.cond.updated, col="black", lwd=1.2*2)
lines(theta.grid,upper,lty=2,col="blue", lwd=1.2*2)
lines(theta.grid,lower,lty=2,col="green", lwd=1.2*2)
points(theta.cond.updated, y.cond.updated, col = "red", pch = 19)
legend(0.35,0.9,legend = c(expression(paste(mu[cond])), "upper prediction bound", "lower prediction bound", "Values conditioned on"),
       col = c("black","blue","green", "red"), cex = 0.8, lty = c(1,2,2,NA), lwd=2.4, pch=c(NA, NA, NA, 19))
dev.off();

threshold <- 0.30;
probs.updated <- pnorm(threshold, mean=mu.cond.updated, sd=sqrt(var.updated));

task <- 'task2c_probs.pdf'

pdf(task)
par(mar=c(5,6,4,1)+.1)
plot(NULL,NULL, xlim = c(0.25,0.5), ylim = c(0, 0.4), main = expression(paste(P(Y(theta)<0.30))),
     xlab = expression(paste(theta)), ylab = expression(paste(P(Y(theta)))), cex.lab = 1.5)
lines(theta.grid, probs.updated, col="blue", lwd=1.2*2)
legend(0.35,0.9,legend = expression(paste(P(Y(theta)<0.30))), cex = 0.8, lty = 1, lwd=2.4, col="blue")
dev.off();

y.best <- theta[match(max(probs.updated),probs.updated)]
print("Best y: ")
print(y.best)
