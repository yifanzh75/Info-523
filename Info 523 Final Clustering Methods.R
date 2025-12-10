# Load libraries
library(readr)
library(dplyr)
library(ggplot2)
library(GGally)
library(caret)
library(plotly)
library(tidyr)
library(scales)
library(reshape2)
library(corrplot)
library(cluster)
library(FNN)

# Set options
options(dplyr.width = Inf)

# Read the data
df <- read.csv("/Users/oliviadodge/Downloads/health_lifestyle_classification.csv")

# Columns to delete
delete_columns <- c(
  'screen_time','family_history','mental_health_score','occupation','mental_health_support',
  'device_usage','healthcare_access','insurance','pet_owner','height','weight','waist_size',
  'bmi_estimated','bmi_scaled','bmi_corrected','physical_activity','education_level','job_type',
  'income','electrolyte_level','gene_marker_flag','environmental_risk_score','daily_supplement_dosage',
  'sleep_hours','sleep_quality','work_hours','daily_steps','calorie_intake','sugar_intake',
  'alcohol_consumption','smoking_level','water_intake','stress_level','diet_type','exercise_type',
  'sunlight_exposure','meals_per_day','caffeine_intake'
)

df <- df %>% select(-all_of(delete_columns))

# Check info
glimpse(df)
colSums(is.na(df))

# ID and target
id_column <- "survey_code"
target_column <- "target"
numerical_columns <- c('age','bmi','blood_pressure','heart_rate','cholesterol','glucose','insulin')
categorical_columns <- c('gender')

# Drop rows with NA in non-numeric columns
non_numeric <- setdiff(names(df), c(numerical_columns, id_column, target_column))
df <- df %>% drop_na(all_of(non_numeric))

# Remove duplicates
cat("Exact takes:", sum(duplicated(df)), "\n")
df <- df %>% distinct()
cat("Duplicate IDs:", sum(duplicated(df[[id_column]])), "\n")
df <- df %>% distinct(across(all_of(id_column)), .keep_all = TRUE)

# Split features and target
X_cols <- setdiff(names(df), c(id_column, target_column))
X <- df %>% select(all_of(X_cols))
y <- df[[target_column]]

# Train-test split
set.seed(26)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Binary target
y_bin <- as.integer(df[[target_column]] == "diseased")

# Fill numeric NAs with median
for(col in numerical_columns) {
  if(col %in% names(X_train)) {
    X_train[[col]][is.na(X_train[[col]])] <- median(X_train[[col]], na.rm = TRUE)
  }
}
colSums(is.na(X_train))

# Outlier detection functions
sigma_bounds <- function(df, cols, k = 3) {
  bounds <- list()
  for(c in cols) {
    s <- df[[c]]
    m <- mean(s, na.rm = TRUE)
    sd_val <- sd(s, na.rm = TRUE)
    bounds[[c]] <- c(m - k*sd_val, m + k*sd_val)
  }
  return(bounds)
}

outlier_indices_by_bounds <- function(df, bounds) {
  out <- integer(0)
  for(c in names(bounds)) {
    if(c %in% names(df)) {
      idx <- which(df[[c]] < bounds[[c]][1] | df[[c]] > bounds[[c]][2])
      out <- union(out, idx)
    }
  }
  return(out)
}

# Remove outliers
bounds <- sigma_bounds(X_train, numerical_columns, k = 3)
out <- outlier_indices_by_bounds(X_train, bounds)
df <- df[-out, ]
X_train <- X_train[-out, ]
y_train <- y_train[-out]

cat(length(X_train) == length(y_train), "\n")

# Correlation heatmaps
cor_spearman <- cor(X_train[numerical_columns], method = "spearman")
cor_pearson <- cor(X_train[numerical_columns], method = "pearson")

# Correlation plots
corrplot(cor_spearman, method = "color", type = "upper")
corrplot(cor_pearson, method = "color", type = "upper")

# Clip outliers in test set
for(col in names(bounds)) {
  if(col %in% names(X_test)) {
    lower <- bounds[[col]][1]
    upper <- bounds[[col]][2]
    X_test[[col]] <- pmin(pmax(X_test[[col]], lower), upper)
  }
}

# Binary target for current df
y_bin <- as.integer(df[[target_column]] == "diseased")

# Correlation with numeric columns
cor_with_y <- sapply(numerical_columns, function(c) cor(df[[c]], y_bin, use="complete.obs"))
cor_with_y <- sort(cor_with_y, decreasing = TRUE)
cor_with_y

# Scatter plots
pairs <- list(
  c('insulin','cholesterol','insulin vs cholesterol'),
  c('glucose','insulin','glucose vs insulin'),
  c('bmi','cholesterol','bmi vs cholesterol'),
  c('heart_rate','blood_pressure','heart rate vs blood pressure')
)

palette <- c('diseased'='red', 'healthy'='blue')
par(mfrow=c(2,2))
for(p in pairs) {
  plot(df[[p[1]]], df[[p[2]]], col = palette[df[[target_column]]], pch=16,
       main=p[3], xlab=p[1], ylab=p[2])
}

# Histograms
ggplot(df, aes(x=heart_rate, fill=target)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
  scale_fill_manual(values = palette)

ggplot(df, aes(x=insulin, fill=target)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
  scale_fill_manual(values = palette)

ggplot(df, aes(x=blood_pressure, fill=target)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
  scale_fill_manual(values = palette)

ggplot(df, aes(x=age, fill=target)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
  scale_fill_manual(values = palette)

ggplot(df, aes(x=bmi, fill=target)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
  scale_fill_manual(values = palette)

ggplot(df, aes(x=cholesterol, fill=target)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
  scale_fill_manual(values = palette)

ggplot(df, aes(x=glucose, fill=target)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
  scale_fill_manual(values = palette)

# HIERARCHICAL CLUSTERING DATA PREP
# Extract numeric + categorical from the *raw, cleaned* df
X_hier <- df %>% select(all_of(c(numerical_columns, categorical_columns)))

# Convert gender to factor (needed for Gower)
X_hier$gender <- factor(X_hier$gender)

# Save for use later
df_full_hier_clean <- X_hier

# MinMax scaling
preProcValues <- preProcess(X_train[numerical_columns], method = c("range"))
X_train[numerical_columns] <- predict(preProcValues, X_train[numerical_columns])
X_test[numerical_columns] <- predict(preProcValues, X_test[numerical_columns])

# One-hot encoding for categorical
ohe <- dummyVars(~., data=X_train[categorical_columns])
X_train_categorical <- predict(ohe, X_train[categorical_columns])
X_test_categorical <- predict(ohe, X_test[categorical_columns])

# Combine numeric and categorical
X_train <- cbind(X_train[numerical_columns], X_train_categorical)
X_test <- cbind(X_test[numerical_columns], X_test_categorical)

# Check
head(X_train)
head(X_test)
str(X_train)


# ----- Imputation (median) -----
# Train numeric columns
X_train_imputed <- X_train %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)))

X_test_imputed <- X_test %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), median(X_train[[cur_column()]], na.rm = TRUE), .x)))

# Combine train + test
df_full <- bind_rows(X_train_imputed, X_test_imputed)
cat(dim(X_train_imputed), dim(X_test_imputed), dim(df_full), "\n")

# Convert target to binary
y_to_bin <- function(y) {
  s <- y
  unique_vals <- unique(na.omit(s))
  
  if(all(unique_vals %in% c(0,1))) {
    return(as.integer(s))
  } else {
    m <- c("healthy"=0, "diseased"=1)
    return(as.integer(m[as.character(s)]))
  }
}

ytr <- y_to_bin(y_train)
names(ytr) <- 'target'
names(ytr) <- rownames(X_train) # Keep rownames aligned

yte <- y_to_bin(y_test)
names(yte) <- 'target'
names(yte) <- rownames(X_test)

# Copies for k means clustering 
df_full_kmeans <- df_full
X_clust_kmeans <- X_clust
X_clust_kmeans <- scale(X_clust)

library(stats)
library(factoextra)

# ----- Elbow Method -----
clustering_score <- numeric()

for(i in 2:10){
  set.seed(26)
  kmeans_model <- kmeans(X_clust_kmeans, centers = i, nstart = 30)
  clustering_score[i-1] <- kmeans_model$tot.withinss
}

# Plot the elbow

elbow_df <- data.frame(k = 2:10, tot_withinss = clustering_score)
ggplot(elbow_df, aes(x = k, y = tot_withinss)) +
  geom_line() + geom_point() +
  ggtitle("The Elbow Method") +
  xlab("Number of Clusters") + ylab("Clustering Score") +
  theme_minimal()

# KMeans with chosen clusters
kmeans_model <- kmeans(X_clust_kmeans, centers = 3, nstart = 30)
pred_kmeans <- kmeans_model$cluster
df_full_kmeans$cluster <- as.integer(pred_kmeans)

# Cluster counts
table(df_full_kmeans$cluster)

# PCA for 2D visualization
pca_kmeans <- prcomp(X_clust_kmeans, scale. = TRUE)
X2 <- pca_kmeans$x[,1:2]  # PCA projection of points
centers2 <- predict(pca_kmeans, newdata = kmeans_model$centers)

# Plot clusters in PCA space
pca_df <- data.frame(X2, cluster = factor(pred_kmeans))
centers_df <- data.frame(centers2, cluster = factor(1:nrow(centers2)))

ggplot(pca_df, aes(x=PC1, y=PC2, color=cluster)) +
  geom_point(alpha=0.5, size=2) +
  geom_point(data=centers_df, aes(x=PC1, y=PC2), shape=4, size=6, stroke=1.5, color="black") +
  ggtitle("KMeans clusters (PCA projection)") +
  theme_minimal()

# PCA loadings
load_pc1 <- sort(abs(pca_kmeans$rotation[,1]), decreasing = TRUE)
load_pc2 <- sort(abs(pca_kmeans$rotation[,2]), decreasing = TRUE)
cat("PC1 top:\n"); print(round(load_pc1[1:10], 3))
cat("PC2 top:\n"); print(round(load_pc2[1:10], 3))

# Cluster statistics 
glob_mean <- colMeans(df_full_kmeans[X_clust_kmeans %>% colnames()])

for(k in sort(unique(df_full_kmeans$cluster))){
  sub <- df_full_kmeans %>% filter(cluster == k)
  top_dev <- sort(abs(colMeans(sub[X_clust_kmeans %>% colnames()]) - glob_mean), decreasing = TRUE)[1:5]
  cat(sprintf("\nCluster %d: size=%d, age_mean=%.3f, bmi_mean=%.3f\n",
              k, nrow(sub), mean(sub$age), mean(sub$bmi)))
  cat("Top deviations:\n")
  print(round(top_dev, 3))
}

# Percentage distribution by sex 
pct_gender <- df_full_kmeans %>%
  group_by(cluster) %>%
  summarise(
    male_pct   = mean(genderMale) * 100,
    female_pct = mean(genderFemale) * 100
  )
print(pct_gender)

# Add target and compute cluster risk
target_full_kmeans <- c(ytr, yte)
df_full_kmeans$target <- as.integer(target_full_kmeans)

risk_kmeans <- df_full_kmeans %>%
  group_by(cluster) %>%
  summarise(target_mean = round(mean(target), 3))
cat("\nProportion of Patients by Cluster:\n")
print(risk_kmeans)

# HIERARCHICAL CLUSTERING
# Sample the dataset for hierarchical clustering
sample_size <- 5000  # adjust as needed
if(nrow(X_clust_hier) > sample_size){
  sampled_idx <- sample(1:nrow(X_clust_hier), sample_size)
} else {
  sampled_idx <- 1:nrow(X_clust_hier)
}

X_clust_sample <- X_clust_hier[sampled_idx, ]
df_full_sample <- df_full_hier[sampled_idx, ]

# Sample for dendrogram
n_samples_for_dendro <- min(1000, nrow(X_clust_sample))
X_sample_dendro <- X_clust_sample[sample(1:nrow(X_clust_sample), n_samples_for_dendro), ]

# Hierarchical clustering linkage (Ward)
linked <- hclust(dist(X_sample_dendro), method = "ward.D2")

# Reset graphics layout to single plot
par(mfrow = c(1, 1))  

# Plot dendrogram
plot(linked, labels = FALSE, hang = -1,
     main = "Dendrogram (sampled)", xlab = "Objects in subcluster", ylab = "Distance")
rect.hclust(linked, k = 2, border = "red")

#  Agglomerative clustering on full sampled data 
hier_labels <- cutree(hclust(dist(X_clust_sample), method="ward.D2"), k=2)
df_full_sample$cluster <- as.integer(hier_labels)

# PCA for 2D visualization 
pca_hier <- prcomp(X_clust_sample, scale.=TRUE)
X2 <- pca_hier$x[,1:2]
pca_df <- data.frame(X2, cluster=factor(hier_labels))

ggplot(pca_df, aes(x=PC1, y=PC2, color=cluster)) +
  geom_point(alpha=0.6, size=2) +
  ggtitle("Clustering PCA Projection") +
  theme_minimal() +
  labs(x="PC1", y="PC2")

# Cluster statistics
glob_mean <- colMeans(df_full_sample[colnames(X_clust_sample)])
for(k in sort(unique(df_full_sample$cluster))){
  sub <- df_full_sample %>% filter(cluster == k)
  top_dev <- sort(abs(colMeans(sub[colnames(X_clust_sample)]) - glob_mean), decreasing=TRUE)[1:5]
  cat(sprintf("\nCluster %d: size=%d, age_mean=%.3f, bmi_mean=%.3f\n",
              k, nrow(sub), mean(sub$age), mean(sub$bmi)))
  cat("Top deviations:\n")
  print(round(top_dev,3))
}

# PCA stuff
hier_pc1 <- sort(abs(pca_hier$rotation[,1]), decreasing = TRUE)
hier_pc2 <- sort(abs(pca_hier$rotation[,2]), decreasing = TRUE)
cat("PC1 top:\n"); print(round(hier_pc1[1:10], 2))
cat("PC2 top:\n"); print(round(hier_pc2[1:10], 2))

library(mclust)


#  Scale features
X_scaled <- scale(X_clust_kmeans) 

# Fit Gaussian Mixture Model
# mclust will select the best number of clusters using BIC
gmm_model <- Mclust(X_scaled)
summary(gmm_model)

# Cluster assignments
pred_gmm <- gmm_model$classification
df_full_gmm <- df_full_kmeans  # copy original df
df_full_gmm$cluster <- as.integer(pred_gmm)

# Cluster counts
print(table(df_full_gmm$cluster))

pca_gmm <- prcomp(X_scaled, scale. = TRUE)
X2 <- pca_gmm$x[, 1:2]  # first 2 PCs

pca_df <- data.frame(X2, cluster = factor(pred_gmm))

# PCA loadings
load_pc1 <- sort(abs(pca_gmm$rotation[,1]), decreasing = TRUE)
load_pc2 <- sort(abs(pca_gmm$rotation[,2]), decreasing = TRUE)
cat("PC1 top:\n"); print(round(load_pc1[1:10], 3))
cat("PC2 top:\n"); print(round(load_pc2[1:10], 3))

# Cluster statistics
glob_mean <- colMeans(df_full_gmm[colnames(X_scaled)])

for(k in sort(unique(df_full_gmm$cluster))){
  sub <- df_full_gmm %>% filter(cluster == k)
  top_dev <- sort(abs(colMeans(sub[colnames(X_scaled)]) - glob_mean), decreasing = TRUE)[1:5]
  cat(sprintf("\nCluster %d: size=%d, age_mean=%.3f, bmi_mean=%.3f\n",
              k, nrow(sub), mean(sub$age), mean(sub$bmi)))
  cat("Top deviations:\n")
  print(round(top_dev, 3))
}
