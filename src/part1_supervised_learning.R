options(repos = c(CRAN = "https://cloud.r-project.org/")) # Set the CRAN mirror
# Step 1: Initial Setup
# Load necessary packages
# Read the datasets for training and test
myStudentData <- read.table(
"data/student.txt",
header = TRUE, stringsAsFactors = TRUE)
myTestData <- read.table(
"data/studentadditional.txt", header = TRUE, stringsAsFactors = TRUE)
str(myStudentData)
str(myTestData)
library(fastDummies)
# Part 1 - Supervised Learning
# Separating features and target variable for both training and test datasets
train_features <- myStudentData[, -which(names(myStudentData) == "class")]
test_features <- myTestData[, -which(names(myTestData) == "class")]
train_class <- as.factor(myStudentData$class)
test_class <- as.factor(myTestData$class)
# Define categorical columns for one-hot encoding
cat_columns <- c("sex", "address", "famsize", "Pstatus", "schoolsup", "famsup",
"paid", "activities", "nursery", "higher", "romantic")
# Apply dummy encoding, keeping only one binary column per categorical variable
train_features_numeric <- dummy_cols(train_features,
select_columns = cat_columns,
remove_selected_columns = TRUE,
remove_first_dummy = TRUE)
test_features_numeric <- dummy_cols(test_features,
select_columns = cat_columns,
remove_selected_columns = TRUE,
remove_first_dummy = TRUE)
train_data <- cbind(train_features_numeric,train_class)
test_data <- cbind(test_features_numeric,test_class)
str(train_data)
str(test_data)
# Normalization of Training Features
train_features_numeric_norm <- train_features_numeric
# Normalize Test Features
test_features_numeric_norm <- test_features_numeric
for(i in 1:ncol(train_features_numeric_norm)) {
if (is.numeric(train_features_numeric_norm[, i]) ||
is.integer(train_features_numeric_norm[, i])) {
minimum <- min(train_features_numeric_norm[, i], na.rm = TRUE)
maximum <- max(train_features_numeric_norm[, i], na.rm = TRUE)
# Normalize training features
train_features_numeric_norm[, i] <- (train_features_numeric_norm[, i] -
minimum) / (maximum - minimum)
test_features_numeric_norm[, i] <- (test_features_numeric_norm[, i] -
minimum) / (maximum - minimum)
# Combine normalized features with class labels, ensuring 'class' remains a factor
train_dataNORM <- as.data.frame(train_features_numeric_norm)
train_dataNORM$class <- factor(train_class)
test_dataNORM <- as.data.frame(test_features_numeric_norm)
test_dataNORM$class <- factor(test_class)
summary(train_dataNORM)
str(train_dataNORM)
summary(test_dataNORM)
str(test_dataNORM)
# Check class distribution
table(train_dataNORM$class)
table(test_dataNORM$class)
#Step 2: Logistic Regression
# Load necessary libraries
library(caret)
#Apply Logistic Regression glm formula
logistic_model <- glm(class ~ ., family = "binomial", data = train_dataNORM)
# Summarize the model to view coefficients and statistics
summary(logistic_model)
#In logistic regression, when used for probability estimation, has not got a natural
#way to measure fit.
#Both, Cox-Snell and Nagelkerke R Square, are a try to simulate what R Square
#does in Linear Regression.
#An alternative way to measure fit is the log likelihood.
#This is always a negative number. When comparing two models in terms of fit,
#we would like to choose the one with the smallest absolute value
#of the log likelihood.
#Stepwise model selection based on AIC
#The step() function performs model selection in a stepwise fashion based
#on the Akaike Information Criterion (AIC).
#This criterion is a tradeoff between the number of variables used and the fit
#of the model obtained with those variables.
LogModelStep <- step(logistic_model)
# Print the summary of the selected model
summary(LogModelStep)
# Predicting class probabilities on the test data
probabilities <- predict(LogModelStep, newdata = test_dataNORM[
, -which(names(test_dataNORM) == "class")], type = "response")
summary(probabilities)
#classes are "High" and "Low
predictions <- ifelse(probabilities > 0.6, "Low", "High")
# Create confusion matrix directly using the class column from the test dataset
classification_tableLR <- table(pred = predictions, Actual = test_dataNORM$class)
print(classification_tableLR)
# Calculate accuracy
acctestLR <- sum(diag(classification_tableLR)) / sum(classification_tableLR)
print(paste("Accuracy:", acctestLR))
# Ensure predictions are factors with the same levels
predictions <- factor(predictions, levels = c("Low", "High"))
# Ensure actual classes are also factors with the same levels
actual_classes <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Performance measurements
library(caret)
conf_matrix <- confusionMatrix(predictions, actual_classes, positive = "High")
print(conf_matrix)
#Step 3: Support Vector Machines
#Installation of Necessary Packages
library(e1071)
#note that the package e1071 gets ‘confused’ if the label given by a numeric column.
#In that case it will return an SVM-regression as opposed to an SVM-classification model.
#To avoid any confusion, make sure that your class labels are not numeric.
#Recall that the e1071 package calls ‘cost’ to the tradeoff parameter C.
#This function reshuffles the dataset, and you may want to set the seed for the
#random generator using the function set.seed().
#Linear SVM
set.seed(1000)
#Tune SVM and find best Value of tradeoff parameter C with a 10-fold cross-validation
tune_linear <- tune.svm(class ~ ., data = train_dataNORM,
cost = 2ˆ(-12:12),
kernel = "linear")
# Save the best model parameters
bestcostlinear <- tune_linear$best.parameters[[1]]
# Summary of cross-validation results
summary(tune_linear)
#Use the best vaue of the parameter to build a new model
best_linearSVM <- svm(class ~ ., data = train_dataNORM,
kernel = "linear",
cost = bestcostlinear)
summary(best_linearSVM)
# Exclude the target variable 'class' from the test dataset
test_features <- test_dataNORM[, -which(names(test_dataNORM) == "class")]
# Predicting: using the best linear SVM model on the test dataset
svm_predictions <- predict(best_linearSVM, newdata = test_features)
# Ensure the predictions and actual classes are factors with the same levels
svm_predictions <- factor(svm_predictions, levels = c("Low", "High"))
actual_classes <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Creating the confusion matrix using the `confusionMatrix()` function
library(caret)
conf_matrix <- confusionMatrix(svm_predictions, actual_classes, positive = "High")
# Print the confusion matrix and associated statistics
print(conf_matrix)
#The hyperplane is written as myomega[1,1]*x1 + myomega[1,2]*x2 ... + myb = 0
#It is important you realize this cannot be done for the RBF kernel, but only for the linear kernel you can.
#RBF
set.seed(1000)
# Tune SVM with RBF kernel and find the best values for cost and gamma with 10-fold cross-validation
tune_rbf <- tune.svm(class ~ ., data = train_dataNORM,
kernel = "radial",
cost = 2ˆ(-12:12), gamma = 2ˆ(-12:12),
tunecontrol = tune.control(cross = 10))
# Save the best model parameters for both cost and gamma
bestcost_rbf <- tune_rbf$best.parameters$cost
bestgamma_rbf <- tune_rbf$best.parameters$gamma
print(bestcost_rbf)
print(bestgamma_rbf)
# Summary of cross-validation results
summary(tune_rbf)
# Use the best value of the parameters to build a new RBF SVM model
best_rbfSVM <- svm(class ~ ., data = train_dataNORM,
kernel = "radial",
cost = bestcost_rbf,
gamma = bestgamma_rbf)
# Summary of the best RBF SVM model
summary(best_rbfSVM)
print(bestcost_rbf)
print(bestgamma_rbf)
# Predicting: using the best RBF SVM model on the test dataset
svm_predictions_rbf <- predict(best_rbfSVM, newdata = test_features)
# Ensure the predictions and actual classes are factors with the same levels
svm_predictions_rbf <- factor(svm_predictions_rbf, levels = c("Low", "High"))
actual_classes_rbf <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Create a confusion matrix to compare predictions with the actual class labels
conf_matrix_rbf <- confusionMatrix(svm_predictions_rbf, actual_classes_rbf, positive = "High")
# Print the confusion matrix and associated statistics
print("Confusion Matrix - Best RBF SVM:")
print(conf_matrix_rbf)
#Step 4: Classification Trees
#Installation of Necessary Packages
install.packages('tree', dependencies = TRUE)
library('tree')
set.seed(1000)
#do we need this??
#setup <- tree.control(nobs=nrow(minihousing), mincut = 25, minsize = 50,mindev=0.01)
#Building a tree
tree_model <- tree(class ~ ., data = train_dataNORM)
plot(tree_model)
text(tree_model, pretty = 0)
summary(tree_model)
#Perform cross-validation to find the best tree size for pruning
set.seed(1000)
mycv_tree <- cv.tree(tree_model, FUN = prune.tree)
#cv.tree() function does not return a tree, but a collection of trees.
#performs an analysis of different levels of pruning, a very aggressive pruning will remove a lot of branches,
#and will construct a very small tree. A very mild pruning will construct a very similar tree.
#Loss Matrix
best_size <- mycv_tree$size[which(mycv_tree$dev==min(mycv_tree$dev))]
#Prune tree using the best size
pruned_tree <- prune.tree(tree_model, best = best_size[1])
plot(pruned_tree)
text(pruned_tree, pretty = 0)
summary(pruned_tree)
library(caret)
library(ggplot2)
library(lattice)
# Exclude the target variable 'class' from the test dataset
test_features <- test_dataNORM[, -which(names(test_dataNORM) == "class")]
#Predict class labels on test dataset using the pruned tree
pruned_predictions <- predict(pruned_tree, newdata = test_features, type='class')
# Create a confusion matrix for the pruned tree predictions
conf_matrix_pruned <- table(Predicted = pruned_predictions, Actual = test_dataNORM$class)
# Print the confusion matrix
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
# Calculate accuracy for the pruned tree
pruned_accuracy <- sum(diag(conf_matrix_pruned)) / sum(conf_matrix_pruned)
print(paste("Accuracy of Pruned Classification Tree:", round(pruned_accuracy, 4)))
# Ensure the predictions and actual classes are factors with the same levels
pruned_predictions <- factor(pruned_predictions, levels = c("Low", "High"))
actual_classes_pruned <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Create a confusion matrix for the pruned tree predictions using the caret package
conf_matrix_pruned <- confusionMatrix(pruned_predictions, actual_classes_pruned, positive = "High")
# Print the confusion matrix and associated statistics
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
install.packages('rpart', dependencies = TRUE)
install.packages("rpart.plot", dependencies = TRUE)
# Step 4: Classification Trees
# Load necessary packages
library('rpart')
library(rpart.plot)
set.seed(1000)
#Try1: ratio 1.85 - 1
# Loss matrix with a higher cost for misclassifying "High"
#If misclassifying a "High" instance (False Negative) is significantly more critical, you might want to increase its cost.
myloss <- matrix(c(0, 1.85, # Cost of True Positive and False Positive
1, 0), # Cost of False Negative and True Negative
39
nrow = 2, byrow = TRUE)
# Build the classification tree using the rpart function
tree_model <- rpart(class ~ ., data = train_dataNORM, parms = list(loss = myloss))
# Plot the tree
# Print the summary of the tree
summary(tree_model)
library(rpart)
library(rpart.plot)
# Find the best CP value with minimum cross-validation error
best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
print(paste("Best CP:", best_cp))
# Prune the tree using the selected best CP value
pruned_tree <- prune(tree_model, cp = best_cp)
# Plot the pruned tree
# Print the summary of the pruned tree
summary(pruned_tree)
library(caret)
library(ggplot2)
library(lattice)
# Exclude the target variable 'class' from the test dataset
test_features <- test_dataNORM[, -which(names(test_dataNORM) == "class")]
# Predict class labels on test dataset using the pruned tree
pruned_predictions <- predict(pruned_tree, newdata = test_features, type = 'class')
# Create a confusion matrix for the pruned tree predictions
conf_matrix_pruned <- table(Predicted = pruned_predictions, Actual = test_dataNORM$class)
# Print the confusion matrix
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
# Calculate accuracy for the pruned tree
pruned_accuracy <- sum(diag(conf_matrix_pruned)) / sum(conf_matrix_pruned)
print(paste("Accuracy of Pruned Classification Tree:", round(pruned_accuracy, 4)))
# Ensure the predictions and actual classes are factors with the same levels
pruned_predictions <- factor(pruned_predictions, levels = c("Low", "High"))
actual_classes_pruned <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Create a confusion matrix for the pruned tree predictions
conf_matrix_pruned <- confusionMatrix(pruned_predictions, actual_classes_pruned, positive = "High")
# Print the confusion matrix and associated statistics
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
# Step 4: Classification Trees
# Load necessary packages
library('rpart')
library(rpart.plot)
set.seed(1000)
#Try 2: ratio 1.6 - 1
# Loss matrix with a higher cost for misclassifying "High"
#If misclassifying a "High" instance (False Negative) is significantly more critical, you might want to increase its cost.
myloss <- matrix(c(0, 1.6, # Cost of True Positive and False Positive
1, 0), # Cost of False Negative and True Negative
nrow = 2, byrow = TRUE)
# Build the classification tree using the rpart function
tree_model <- rpart(class ~ ., data = train_dataNORM, parms = list(loss = myloss))
# Plot the tree
# Print the summary of the tree
summary(tree_model)
library(rpart)
library(rpart.plot)
# Find the best CP value with minimum cross-validation error
best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
print(paste("Best CP:", best_cp))
# Prune the tree using the selected best CP value
pruned_tree <- prune(tree_model, cp = best_cp)
# Plot the pruned tree
# Print the summary of the pruned tree
summary(pruned_tree)
library(caret)
library(ggplot2)
library(lattice)
# Exclude the target variable 'class' from the test dataset
test_features <- test_dataNORM[, -which(names(test_dataNORM) == "class")]
# Predict class labels on test dataset using the pruned tree
pruned_predictions <- predict(pruned_tree, newdata = test_features, type = 'class')
# Create a confusion matrix for the pruned tree predictions
conf_matrix_pruned <- table(Predicted = pruned_predictions, Actual = test_dataNORM$class)
# Print the confusion matrix
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
# Calculate accuracy for the pruned tree
pruned_accuracy <- sum(diag(conf_matrix_pruned)) / sum(conf_matrix_pruned)
print(paste("Accuracy of Pruned Classification Tree:", round(pruned_accuracy, 4)))
# Ensure the predictions and actual classes are factors with the same levels
pruned_predictions <- factor(pruned_predictions, levels = c("Low", "High"))
actual_classes_pruned <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Create a confusion matrix for the pruned tree predictions using the caret package
conf_matrix_pruned <- confusionMatrix(pruned_predictions, actual_classes_pruned, positive = "High")
# Print the confusion matrix and associated statistics
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
# Step 4: Classification Trees
# Load necessary packages
library('rpart')
library(rpart.plot)
set.seed(1000)
#Try 3: 1.5-1 ratio
# Loss matrix with a higher cost for misclassifying "High"
#If misclassifying a "High" instance (False Negative) is significantly more critical, you might want to increase its cost.
myloss <- matrix(c(0, 1.5, # Cost of True Positive and False Positive
1, 0), # Cost of False Negative and True Negative
nrow = 2, byrow = TRUE)
# Build the classification tree using the rpart function
tree_model <- rpart(class ~ ., data = train_dataNORM, parms = list(loss = myloss))
# Plot the tree
# Print the summary of the tree
summary(tree_model)
library(rpart)
library(rpart.plot)
# Find the best CP value with minimum cross-validation error
best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
print(paste("Best CP:", best_cp))
# Prune the tree using the selected best CP value
pruned_tree <- prune(tree_model, cp = best_cp)
# Plot the pruned tree
# Print the summary of the pruned tree
summary(pruned_tree)
library(caret)
# Exclude the target variable 'class' from the test dataset
test_features <- test_dataNORM[, -which(names(test_dataNORM) == "class")]
# Predict class labels on test dataset using the pruned tree
pruned_predictions <- predict(pruned_tree, newdata = test_features, type = 'class')
# Create a confusion matrix for the pruned tree predictions
conf_matrix_pruned <- table(Predicted = pruned_predictions, Actual = test_dataNORM$class)
# Print the confusion matrix
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
# Calculate accuracy for the pruned tree
pruned_accuracy <- sum(diag(conf_matrix_pruned)) / sum(conf_matrix_pruned)
print(paste("Accuracy of Pruned Classification Tree:", round(pruned_accuracy, 4)))
# Ensure the predictions and actual classes are factors with the same levels
pruned_predictions <- factor(pruned_predictions, levels = c("Low", "High"))
actual_classes_pruned <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Create a confusion matrix for the pruned tree predictions using the caret package
conf_matrix_pruned <- confusionMatrix(pruned_predictions, actual_classes_pruned, positive = "High")
# Print the confusion matrix and associated statistics
print("Confusion Matrix - Pruned Classification Tree:")
print(conf_matrix_pruned)
#Installation of Necessary Packages
install.packages('randomForest',dependencies=TRUE)
#Step 5: Random Forest
library('randomForest')
set.seed(1000)
#The default value of mtry is squared root of the number of explanatory variables and rounded down.
#This is the parameter that controls how many explanatory variables we can use in each branch node.
#Once mtry is fixed, the set of explanatory variables available at a branch node is chosen randomly.
# Create the Random Forest model
rf_model <- randomForest(class ~ ., data = train_dataNORM,
ntree = 500,
mtry = floor(sqrt(ncol(train_features))),
importance = TRUE)
print(rf_model)
library(caret)
# Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = test_features, type = 'class')
# Create a confusion matrix
conf_matrix_rf <- table(Predicted = rf_predictions, Actual = test_dataNORM$class)
# Print the confusion matrix
print("Confusion Matrix - Random Forest:")
print(conf_matrix_rf)
# Calculate the accuracy of the model
rf_accuracy <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Accuracy of Random Forest:", round(rf_accuracy, 4)))
# Ensure the predictions and actual classes are factors with the same levels
rf_predictions <- factor(rf_predictions, levels = c("Low", "High"))
actual_classes_rf <- factor(test_dataNORM$class, levels = c("Low", "High"))
# Create a confusion matrix for the Random Forest predictions using the caret package
conf_matrix_rf <- confusionMatrix(rf_predictions, actual_classes_rf, positive = "High")
# Print the confusion matrix and associated statistics
print("Confusion Matrix - Random Forest:")
print(conf_matrix_rf)
#Plot importance of features
#Due to imbalance in dataset (low Sensitivity rate) we implement class weights
library(randomForest)
set.seed(1000)
# Specify the class weights
class_weights <- c(High = 0.3, Low = 1)
# Create the Random Forest model with class weights
rf_model_weights <- randomForest(class ~ ., data = train_dataNORM,
mtry = floor(sqrt(ncol(train_dataNORM))),
importance = TRUE,
classwt = class_weights)
# Print the model summary
print(rf_model_weights)
# Make predictions on the test data
rf_prediction <- predict(rf_model_weights, test_dataNORM, type = "class")
# Create the confusion matrix
classification_table <- table(rf_prediction, test_dataNORM$class)
# Calculate accuracy
accuracy_RF_weights <- sum(diag(classification_table)) / sum(classification_table)
print(paste("Accuracy with Weights:", accuracy_RF_weights))
# Performance measurements
library(caret)
RF_Weights_confusion_summary <- confusionMatrix(rf_prediction, test_dataNORM$class, positive = "High")
print(RF_Weights_confusion_summary)
#Plot importance of features
#Due to imbalance in dataset (low Sensitivity rate) we implement class weights
library(randomForest)
set.seed(1000)
# Specify the class weights
class_weights <- c(High = 0.28, Low = 1)
# Create the Random Forest model with class weights
rf_model_weights <- randomForest(class ~ ., data = train_dataNORM,
mtry = floor(sqrt(ncol(train_dataNORM))),
importance = TRUE,
classwt = class_weights)
# Print the model summary
print(rf_model_weights)
# Make predictions on the test data
rf_prediction <- predict(rf_model_weights, test_dataNORM, type = "class")
# Create the confusion matrix
classification_table <- table(rf_prediction, test_dataNORM$class)
# Calculate accuracy
accuracy_RF_weights <- sum(diag(classification_table)) / sum(classification_table)
print(paste("Accuracy with Weights:", accuracy_RF_weights))
# Performance measurements
library(caret)
RF_Weights_confusion_summary <- confusionMatrix(rf_prediction, test_dataNORM$class, positive = "High")
print(RF_Weights_confusion_summary)
# Load necessary package
install.packages("FNN", dependencies = TRUE)
# Step 6: K-Nearest Neighbors
library(FNN)
set.seed(1000)
# Separate features and class labels
train_features <- train_dataNORM[, -which(names(train_dataNORM) == "class")]
test_features <- test_dataNORM[, -which(names(test_dataNORM) == "class")]
train_class <- train_dataNORM$class
test_class <- test_dataNORM$class
# Initialize variables to track the best k and accuracy
bestk <- 0
bestaccuracy <- 0
accuracy <- NULL
# Perform leave-one-out cross-validation for k values from 1 to 15
for (auxk in 1:15) {
knn_cv <- knn.cv(train = train_features, cl = train_class, k = auxk)
cv_table <- table(knn_cv, train_class)
accuracy[auxk] <- sum(diag(cv_table)) / sum(cv_table) # Calculate accuracy
# Update best k and accuracy
if (bestaccuracy < accuracy[auxk]) {
bestk <- auxk
bestaccuracy <- accuracy[auxk]
# Plot accuracy vs. k
plot(accuracy, xlab = "K", ylab = "Cross-validated Accuracy", type = "b")
print(paste("Best k:", bestk))
print(paste("Best Accuracy:", round(bestaccuracy, 4)))
# Use the best k value found from cross-validation
knn_predictions <- knn(train = train_features,
test = test_features,
cl = train_class,
k = bestk)
# Create a confusion matrix to compare predictions with actual labels
conf_matrix_knn <- table(Predicted = knn_predictions, Actual = test_class)
# Print the confusion matrix
print("Confusion Matrix - k-NN (Best k):")
print(conf_matrix_knn)
# Calculate accuracy of the model
knn_accuracy <- sum(diag(conf_matrix_knn)) / sum(conf_matrix_knn)
print(paste("Accuracy of k-NN with Best k:", round(knn_accuracy, 4)))
# Ensure the predictions and actual classes are factors with the same levels
knn_predictions <- factor(knn_predictions, levels = c("Low", "High"))
actual_classes_knn <- factor(test_class, levels = c("Low", "High"))
# Create a confusion matrix for the k-NN predictions using the caret package
conf_matrix_knn <- confusionMatrix(knn_predictions, actual_classes_knn, positive = "High")
# Print the confusion matrix and associated statistics
print("Confusion Matrix - k-NN (Best k):")
print(conf_matrix_knn)
