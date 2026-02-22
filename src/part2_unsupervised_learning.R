#Part 2 - Unsupervised Learning
#Step 1: Setup of the Data
#Load Data
#library(fastDummies)
wine_data <- read.table("data/wine.txt", header = TRUE, stringsAsFactors = TRUE)
#Part 2 - Unsupervised Learning
#Step 1: Setup of the Data
# Standardize the data
mean <- apply(wine_data, 2, mean)
std <- apply(wine_data, 2, sd)
wine_data_standardized <- scale(wine_data, center = mean, scale = std)
summary(wine_data_standardized)
str(wine_data_standardized)
#Distance matrix for standardized data
mydistmatrixWine <- dist(wine_data_standardized)
# Print the distance matrix or a submatrix
mydistsubmatrixSTAN <- as.matrix(mydistmatrixWine)
mydistsubmatrixSTAN <- mydistsubmatrixSTAN[1:10, 1:10] # First 10x10 submatrix
print(mydistsubmatrixSTAN)
#A distance matrix is a square matrix that shows the distance between each pair of #observations
#in your dataset.
#The rows and columns correspond to the observations, and #the values
#in the matrix represent the distances between those observations.
#Step 2: Hierarchical Clustering with different Distances
# Euclidean distance with different linkages
#euclidian distance is default option
#‘hang=-1’ option just changes the position of the labels of the objects.
#They will be placed all of them at the bottom of the dendrogram and at the same level.
hclustEC <- hclust(dist(wine_data_standardized), method = "complete")
plot(hclustEC, hang = -1, cex = 0.5, main = "Cluster - Euclidean Distance, Complete Linkage")
hclust (*, "complete")
dist(wine_data_standardized)
hclustES <- hclust(dist(wine_data_standardized), method = "single")
plot(hclustES, hang = -1, cex = 0.5, main = "Cluster - Euclidean Distance, Single Linkage")
hclust (*, "single")
dist(wine_data_standardized)
hclustEA <- hclust(dist(wine_data_standardized), method = "average")
plot(hclustEA, hang = -1, cex = 0.5, main = "Cluster - Euclidean Distance, Average Linkage")
hclust (*, "average")
dist(wine_data_standardized)
# Maximum distance with different linkages
hclustMaxC <- hclust(dist(wine_data_standardized, method = "maximum"), method = "complete")
plot(hclustMaxC, hang = -1, cex = 0.5, main = "Cluster - Maximum Distance, Complete Linkage")
hclust (*, "complete")
dist(wine_data_standardized, method = "maximum")
hclustMaxS <- hclust(dist(wine_data_standardized, method = "maximum"), method = "single")
plot(hclustMaxS, hang = -1, cex = 0.5, main = "Cluster - Maximum Distance, Single Linkage")
hclust (*, "single")
dist(wine_data_standardized, method = "maximum")
hclustMaxA <- hclust(dist(wine_data_standardized, method = "maximum"), method = "average")
plot(hclustMaxA, hang = -1, cex = 0.5, main = "Cluster - Maximum Distance, Average Linkage")
hclust (*, "average")
dist(wine_data_standardized, method = "maximum")
# Manhattan distance with different linkages
hclustManC <- hclust(dist(wine_data_standardized, method = "manhattan"), method = "complete")
plot(hclustManC, hang = -1, cex = 0.5, main = "Cluster - Manhattan Distance, Complete Linkage")
hclust (*, "complete")
dist(wine_data_standardized, method = "manhattan")
hclustManS <- hclust(dist(wine_data_standardized, method = "manhattan"), method = "single")
plot(hclustManS, hang = -1, cex = 0.5, main = "Cluster - Manhattan Distance, Single Linkage")
hclust (*, "single")
dist(wine_data_standardized, method = "manhattan")
hclustManA <- hclust(dist(wine_data_standardized, method = "manhattan"), method = "average")
plot(hclustManA, hang = -1, cex = 0.5, main = "Cluster - Manhattan Distance, Average Linkage")
hclust (*, "average")
dist(wine_data_standardized, method = "manhattan")
# Set up a 3x3 plotting area for visualizations
# 1. Euclidean Distance with Different Linkages
hclustEC <- hclust(dist(wine_data_standardized), method = "complete")
plot(hclustEC, hang = -1, cex = 0.5, main = "Euclidean, Complete Linkage")
rect.hclust(hclustEC, k = 3)
hclust (*, "complete")
dist(wine_data_standardized)
hclustES <- hclust(dist(wine_data_standardized), method = "single")
plot(hclustES, hang = -1, cex = 0.5, main = "Euclidean, Single Linkage")
rect.hclust(hclustES, k = 3)
hclust (*, "single")
dist(wine_data_standardized)
hclustEA <- hclust(dist(wine_data_standardized), method = "average")
plot(hclustEA, hang = -1, cex = 0.5, main = "Euclidean, Average Linkage")
rect.hclust(hclustEA, k = 3)
hclust (*, "average")
dist(wine_data_standardized)
# 2. Maximum Distance with Different Linkages
hclustMaxC <- hclust(dist(wine_data_standardized, method = "maximum"), method = "complete")
plot(hclustMaxC, hang = -1, cex = 0.5, main = "Maximum, Complete Linkage")
rect.hclust(hclustMaxC, k = 3)
hclust (*, "complete")
dist(wine_data_standardized, method = "maximum")
hclustMaxS <- hclust(dist(wine_data_standardized, method = "maximum"), method = "single")
plot(hclustMaxS, hang = -1, cex = 0.5, main = "Maximum, Single Linkage")
rect.hclust(hclustMaxS, k = 3)
hclust (*, "single")
dist(wine_data_standardized, method = "maximum")
hclustMaxA <- hclust(dist(wine_data_standardized, method = "maximum"), method = "average")
plot(hclustMaxA, hang = -1, cex = 0.5, main = "Maximum, Average Linkage")
rect.hclust(hclustMaxA, k = 3)
hclust (*, "average")
dist(wine_data_standardized, method = "maximum")
# 3. Manhattan Distance with Different Linkages
hclustManC <- hclust(dist(wine_data_standardized, method = "manhattan"), method = "complete")
plot(hclustManC, hang = -1, cex = 0.5, main = "Manhattan, Complete Linkage")
rect.hclust(hclustManC, k = 3)
hclust (*, "complete")
dist(wine_data_standardized, method = "manhattan")
hclustManS <- hclust(dist(wine_data_standardized, method = "manhattan"), method = "single")
plot(hclustManS, hang = -1, cex = 0.5, main = "Manhattan, Single Linkage")
rect.hclust(hclustManS, k = 3)
hclust (*, "single")
dist(wine_data_standardized, method = "manhattan")
hclustManA <- hclust(dist(wine_data_standardized, method = "manhattan"), method = "average")
plot(hclustManA, hang = -1, cex = 0.5, main = "Manhattan, Average Linkage")
rect.hclust(hclustManA, k = 3)
hclust (*, "average")
dist(wine_data_standardized, method = "manhattan")
#Step3: KMeans
# Initialize an empty vector to store the total within-cluster sum of squares for each k
#‘myKmeans3$tot.withinss’ is the function that we are optimizing.
#choose for the clustering with the smallest ‘myKmeans3$tot.withinss’.
mytotwithinss <- NULL
# Loop over values of k from 2 to 15
for (auxk in 2:15) {
# Set the seed at the beginning of each loop iteration for reproducibility
set.seed(1)
# Run K-means with nstart=50 to ensure better convergence
myKmeansauxkmultistart <- kmeans(wine_data_standardized, centers = auxk, nstart = 50)
# Store the total within-cluster sum of squares for each k
mytotwithinss[auxk] <- myKmeansauxkmultistart$tot.withinss
# Plot the total within-cluster sum of squares against the number of clusters k
plot(2:15, mytotwithinss[2:15], type = "b", xlab = "Number of Clusters (k)",
ylab = "Total Within-Cluster Sum of Squares",
main = "Elbow Method for Optimal K")
#A common thing to do is to choose the K using the elbow rule.
#Now we choose the best k --> in our case = 4
# Set the seed for reproducibility
set.seed(1)
# Perform K-means clustering with the chosen optimal k (here is 4)
optimal_k <- 4
myKmeans_optimal <- kmeans(wine_data_standardized, centers = optimal_k, nstart = 50)
# Convert the standardized data to a data frame for plotting
wine_data_df <- as.data.frame(wine_data_standardized)
# Plot the pairwise scatter plots with clusters as colors for all variables
with(wine_data_df, pairs(wine_data_df, col = c(1:optimal_k)[myKmeans_optimal$cluster],
main = paste("K-means Clustering with K =", optimal_k)))
# Plot the pairwise scatter plots for the first two variables only, with clusters as colors
with(wine_data_df, pairs(wine_data_df[, 1:2], col = c(1:optimal_k)[myKmeans_optimal$cluster],
main = paste("K-means Clustering (First Two Variables) with K =", optimal_k)))
