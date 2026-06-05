# Load libraries
library(readxl)
library(stringr)
library(tm)
library(SnowballC)
library(caret)
library(e1071)
library(class)

# Load dataset
setwd("C:/Users/HP/OneDrive/Desktop/Masters/SpamEmailDetection_using_cart")
data <- read_excel("SMSSpamCollection.xlsx")
msg <- data.frame(type = data[[1]], message = data[[2]])
colnames(msg) <- c("type", "message")
msg$type <- as.factor(msg$type)
msg$token <- as.numeric(msg$type == "spam")

# Preprocessing using VCorpus
corpus <- VCorpus(VectorSource(msg$message))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)

# Create Document-Term Matrix and remove sparse terms
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.995)
msg_matrix_stem <- as.matrix(dtm)

# Split dataset into training and test sets
set.seed(20)
train_index <- createDataPartition(y = msg$token, p = 0.5, list = FALSE)
msg_trained_stem <- msg_matrix_stem[train_index, ]
msg_test_stem <- msg_matrix_stem[-train_index, ]
type_trained_stem <- msg$token[train_index]
type_test_stem <- msg$token[-train_index]

# Naive Bayes model (convert to factor columns so e1071 uses categorical NB with
# Laplace smoothing instead of Gaussian NB, which breaks on zero-variance features)
type_trained_factor <- factor(type_trained_stem)
to_factor_df <- function(mat) {
  as.data.frame(lapply(as.data.frame((mat > 0) * 1L),
                       function(x) factor(x, levels = c(0L, 1L))))
}
msg_trained_nb <- to_factor_df(msg_trained_stem)
msg_test_nb    <- to_factor_df(msg_test_stem)
model1 <- naiveBayes(x = msg_trained_nb, y = type_trained_factor, laplace = 1)
pred1 <- predict(model1, newdata = msg_test_nb)
model1_accuracy <- mean(as.numeric(as.character(pred1)) == type_test_stem)

# Logistic Regression model
model2 <- glm(type_trained_stem ~ ., data = as.data.frame(msg_trained_stem), family = binomial)
pred2 <- predict(model2, newdata = as.data.frame(msg_test_stem), type = "response")
pred2_binary <- ifelse(pred2 > 0.5, 1, 0)
model2_accuracy <- mean(pred2_binary == type_test_stem)

# KNN with k = 5
model3 <- knn(train = msg_trained_stem, test = msg_test_stem, cl = type_trained_stem, k = 5)
pred3 <- as.numeric(as.character(model3))
model3_accuracy <- mean(pred3 == type_test_stem)

# KNN with k = 15
model4 <- knn(train = msg_trained_stem, test = msg_test_stem, cl = type_trained_stem, k = 15)
pred4 <- as.numeric(as.character(model4))
model4_accuracy <- mean(pred4 == type_test_stem)

# Print accuracy of all models
cat("Model accuracy using Naive Bayes: ", model1_accuracy, "\n")
cat("Model accuracy using Logistic Regression: ", model2_accuracy, "\n")
cat("Model accuracy using KNN with k = 5: ", model3_accuracy, "\n")
cat("Model accuracy using KNN with k = 15: ", model4_accuracy, "\n")

