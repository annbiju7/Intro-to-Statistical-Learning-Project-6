setwd("C:/ann/fall 2023/stat 4360")

# Question 1a

# Install and load required libraries
install.packages("ISLR")
library(ISLR)
install.packages("tree")
install.packages("DAAG")
library(tree)
library(DAAG)

#Hitters dataset
data("Hitters")

# Remove rows with missing data
Hitters <- na.omit(Hitters)

# Extract predictors and response
predictors <- hitters_data <- Hitters[, -which(names(Hitters) == "Salary")]
response <- log(Hitters$Salary)

# (a) Fit a tree to the data
tree_model <- tree(response ~ ., data = predictors)

# Display the tree graphically
plot(tree_model)
text(tree_model, pretty = 0)

# Summarize the tree
summary(tree_model)

# Predict and calculate LOOCV
tree_pred <- predict(tree_model, newdata = predictors)
tree_loocv_mse <- mean((tree_pred - response)^2)
print(paste("Estimated test MSE for un-pruned tree:", tree_loocv_mse))

#Question 1b
# (b) Use LOOCV to determine whether pruning is helpful
cv_tree <- cv.tree(tree_model, FUN = prune.tree)
best_tree_size <- cv_tree$size[which.min(cv_tree$dev)]

# Prune the tree with the optimal size
pruned_tree_model <- prune.tree(tree_model, best = best_tree_size)

# Display the pruned tree graphically
plot(pruned_tree_model)
text(pruned_tree_model, pretty = 0)

# Summarize the pruned tree
summary(pruned_tree_model)

# Predict and calculate LOOCV for the pruned tree
pruned_tree_pred <- predict(pruned_tree_model, newdata = predictors)
pruned_tree_loocv_mse <- mean((pruned_tree_pred - response)^2)
print(paste("Estimated test MSE for pruned tree:", pruned_tree_loocv_mse))

# Compare the best pruned and un-pruned trees
print(paste("Optimal size for pruned tree:", best_tree_size))

# Question 1c
install.packages("randomForest")
library(randomForest)

bagging_model <- randomForest(response ~ ., data = predictors, ntree = 1000)

# Predict and calculate LOOCV for bagging
bagging_pred <- predict(bagging_model, newdata = predictors)
bagging_loocv_mse <- mean((bagging_pred - response)^2)
print(paste("Estimated test MSE for bagging (B=1000):", bagging_loocv_mse))

var_importance_bagging <- importance(bagging_model)
print(var_importance_bagging)

# Question 1d
num_predictors <- ncol(predictors)
mtry_value <- round(num_predictors / 3)

rf_model <- randomForest(response ~ ., data = predictors, ntree = 1000, mtry = mtry_value)

# Predict and calculate LOOCV for random forest
rf_pred <- predict(rf_model, newdata = predictors)
rf_loocv_mse <- mean((rf_pred - response)^2)
print(paste("Estimated test MSE for random forest (B=1000, m=p/3):", rf_loocv_mse))

# Display variable importance for random forest
var_importance_rf <- importance(rf_model)
print(var_importance_rf)

# Question 1e
install.packages("gbm")
library(gbm)

boost_model <- gbm(response ~ ., data = predictors, distribution = "gaussian", n.trees = 1000, interaction.depth = 1, shrinkage = 0.01)

# Predict and calculate LOOCV for boosting
boost_pred <- predict(boost_model, newdata = predictors, n.trees = 1000)
boost_loocv_mse <- mean((boost_pred - response)^2)
print(paste("Estimated test MSE for boosting (B=1000, d=1, λ=0.01):", boost_loocv_mse))

# Display variable importance for boosting
var_importance_boost <- summary(boost_model)
print(var_importance_boost)

# Question 2a

#load diabetes data
diabetes <- read.csv("C:\\ann\\fall 2023\\stat 4360\\project 6\\diabetes(1).csv")
diabetes

install.packages("e1071")
install.packages("caret")
library(e1071)
library(caret)


# Set the seed for reproducibility
set.seed(123)

# Extract predictors and response
predictors <- diabetes[, -9]  # Assuming the last column is the Outcome variable
response <- diabetes[, 9]

svmfit <- svm(Outcome ~ ., data = diabetes, kernel = "linear", cost = 0.1, scale = FALSE)
svmfit$index
summary(svmfit)

# Get the optimal cost parameter
optimal_cost <- svm_tune$best.parameters$cost

# Fit the support vector classifier with the optimal cost
svm_model <- svm(Outcome ~ ., data = diabetes, kernel = "linear", cost = 0.1, scale = FALSE)

# Summarize key features of the fit
summary(svm_model)

# Compute estimated test error rate using 10-fold cross-validation
cv_results <- train(Outcome ~ ., data = diabetes, method = "svmLinear", trControl = trainControl(method = "cv", number = 10))
cv_error_rate <- 1 - cv_results$Accuracy

# Question 2b

svmfit_poly <- svm(Outcome ~ ., data = diabetes, kernel = "polynomial", degree = 2, cost = 0.1, scale = FALSE)
summary(svmfit_poly)

# Compute estimated test error rate using 10-fold cross-validation
svmfit_poly <- train(Outcome ~ ., data = diabetes, method = "svmPoly", trControl = trainControl(method = "cv", number = 10))
svmfit_poly <- 1 - cv_results_poly$results$Accuracy
print(paste("Estimated test error rate for polynomial SVM:", cv_error_rate_poly))


# Question 2c
svmfit_radial <- svm(Outcome ~ ., data = diabetes, kernel = "radial", gamma = 1, cost = 0.1, scale = FALSE)
summary(svmfit_radial)


# Question 3c
install.packages("cluster")
library(cluster)

# Standardize the variables
Hitters$League <- as.numeric(as.factor(Hitters$League))
Hitters$Division <- as.numeric(as.factor(Hitters$Division))
Hitters$NewLeague <- as.numeric(as.factor(Hitters$NewLeague))

standardized_predictors <- scale(Hitters)

# Hierarchical clustering with complete linkage and Euclidean distance
hclust_model <- hclust(dist(standardized_predictors), method = "complete")

# Cut the dendrogram to form two clusters
cut_height <- 10
clusters <- cutree(hclust_model, h = cut_height)

# Summarize the cluster-specific means of the variables
cluster_means <- aggregate(standardized_predictors, by = list(cluster = clusters), FUN = mean)
print(cluster_means)

# Summarize the mean salaries of the players in the two clusters
salary_summary <- aggregate(Hitters$Salary, by = list(cluster = clusters), FUN = mean)
colnames(salary_summary) <- c("Cluster", "Mean_Salary")
print(salary_summary)

# Plot the dendrogram
plot(hclust_model, main = "Dendrogram of Hierarchical Clustering", xlab = "Players", sub = "")
rect.hclust(hclust_model, k = 2, border = 2:3)  # Highlight the two clusters with different colors

# Question 3d

# K-means clustering with K = 2
kmeans_model <- kmeans(standardized_predictors, centers = 2, nstart = 20)

# Assign cluster labels to the original data
clusters <- kmeans_model$cluster

# Summarize the cluster-specific means of the variables
cluster_means <- aggregate(standardized_predictors, by = list(cluster = clusters), FUN = mean)
print(cluster_means)

# Summarize the mean salaries of the players in the two clusters
salary_summary <- aggregate(Hitters$Salary, by = list(cluster = clusters), FUN = mean)
colnames(salary_summary) <- c("Cluster", "Mean_Salary")
print(salary_summary)





