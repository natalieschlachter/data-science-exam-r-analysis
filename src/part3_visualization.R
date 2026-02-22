options(repos = c(CRAN = "https://cloud.r-project.org/")) # Set the CRAN mirror
#Step 1:DataPrep Part3
#Load necessary Libraries & dataset
install.packages('scatterplot3d',dependencies=TRUE)
#Part 3 - Visualization
library('scatterplot3d')
seeds_data <- read.table("data/seeds.txt", header = TRUE, stringsAsFactor = TRUE)
#Inspect data
summary(seeds_data)
#Step 2: PCA
#Calculate covariance matrix
covmat_seeds <- cov(seeds_data)
print(round(covmat_seeds, 2))
#Perform PCA
#Note: PCA is sensitive to different scales, so it's essential to standardize the data
#use the options ‘center=T,scale.=T’ within the ‘prcomp()’ function
seeds_pca <- prcomp(seeds_data, center = T, scale. = T)
#Plot the first two Principal Components
x <- seeds_pca$x[,1]
y <- seeds_pca$x[,2]
plot(x, y, xlab = "PC1", ylab = "PC2", main = "Principal Component Analysis")
text(x, y, labels = row.names(seeds_data), cex = 0.5, pos = 4)
#Plot first three Principal Components in 3D
x <- seeds_pca$x[,1]
y <- seeds_pca$x[,2]
z <- seeds_pca$x[,3]
scatterplot3d(x, y, z, xlab = "PC1", ylab = "PC2", zlab = "PC3", main = "Principal Component Analysis in 3D")
text(x, y, z, labels = row.names(seeds_data), cex = 0.5)
#Print the covariance matrix of the Principal Components, and sum the diagonal
covmat_pca <- cov(seeds_pca$x)
print(round(covmat_pca, 2))
sum_diag <- sum(diag(covmat_pca))
print(paste("Sum of the diagonal of PCA covariance matrix:", sum_diag))
#Print information on the proportion of total variance explained by the Principal Components
summary(seeds_pca)
install.packages("factoextra")
# Load the package
library(factoextra)
library(ggplot2)
#Visualize the explained variance
#Step 3: MultiDimensional Scaling
# Normalization scales data between 0 and 1
min_vals <- apply(seeds_data, 2, min)
max_vals <- apply(seeds_data, 2, max)
seeds_norm <- scale(seeds_data, center = min_vals, scale = max_vals - min_vals)
seeds_norm_df <- as.data.frame(seeds_norm)
summary(seeds_norm_df)
str(seeds_norm_df)
#Perform MDS with nromalized data with Euclidean distance, setting dimensions to 2
seeds_mds <- cmdscale(dist(seeds_norm_df), k = 2, eig = TRUE)
#Plot the MDS result
x <- seeds_mds$points[,1]
y <- seeds_mds$points[,2]
plot(x, y, xlab = "Representative's Coordinate 1", ylab = "Representative's Coordinate 2",
main = "MDS (Euclidean Distance)")
text(x, y, labels = row.names(seeds_norm_df), cex = 0.7)
#Experiment with other distance metrics
#Manhattan distance
seeds_mds_manhattan <- cmdscale(dist(seeds_norm_df, method = "manhattan"),
k = 2, eig = TRUE)
x_manhattan <- seeds_mds_manhattan$points[,1]
y_manhattan <- seeds_mds_manhattan$points[,2]
plot(x_manhattan, y_manhattan, xlab = "Representative's Coordinate 1", ylab = "Representative's Coordinate 2", main = "MDS (Manhattan Distance)")
text(x_manhattan, y_manhattan, labels = row.names(seeds_norm_df), cex = 0.7)
#Maximum distance
seeds_mds_maximum <- cmdscale(dist(seeds_norm_df, method = "maximum"), k = 2, eig = TRUE)
x_maximum <- seeds_mds_maximum$points[,1]
y_maximum <- seeds_mds_maximum$points[,2]
plot(x_maximum, y_maximum, xlab = "Representative's Coordinate 1", ylab = "Representative's Coordinate 2", main = "MDS (Maximum Distance)")
text(x_maximum, y_maximum, labels = row.names(seeds_norm_df), cex = 0.7)
