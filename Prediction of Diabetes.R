# Random Forest Analysis for Diabetes Data Set

str(Diabetes)
attach(Diabetes)

table(Class.variable)

round(prop.table(table(Class.variable))*100, 1)


#Create a function to normalize the data
normFunc <- function(x){(x-mean(x, na.rm = T))/sd(x, na.rm = T)}
Diabetes[1:8] <- apply(Diabetes[1:8], 2, normFunc)
View(Diabetes)

Diabetes_rand <- Diabetes[order(runif(768)), ]
View(Diabetes_rand)

# create training and test data
train <- Diabetes_rand[1:500, ]
test <- Diabetes_rand[501:768, ]


# Building a random forest model on training data 
install.packages("randomForest")
library(randomForest)

?randomForest
Diabetes_forest <- randomForest(train$Class.variable ~ ., data = train, importance = TRUE)
plot(Diabetes_forest)

# Test Data Accuracy
test_acc <- mean(test$Class.variable == predict(Diabetes_forest, newdata=test))
test_acc # 0.776

# Train Data Accuracy
train_acc <- mean(train$Class.variable == predict(Diabetes_forest, data=train))
train_acc # 0.76
              


