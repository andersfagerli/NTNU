A <- matrix(c(9, -2,-2, 6), nrow = 2, ncol = 2)
A
B <- t(A)
B
P <- eigen(A)
eigenval <- P$values
eigenvec <- P$vectors
eigen_matrix <- matrix(c(eigenval[1],0,0,eigenval[2]), nrow = 2, ncol = 2)
diag_A <- eigenvec%*%eigen_matrix%*%t(eigenvec)
corr_A <- cov2cor(A)

E_x <- c(3,1)
Cov_x = A

C <- matrix(c(1,1,1,2), nrow = 2, ncol = 2)
E_c = C%*%E_x