library(ggplot2)
pca <- prcomp(USArrests, scale = TRUE)

##a
#pca$rotation

# manually
corrmatrix <- cor(USArrests)
corrmatrix
eigen(corrmatrix)
