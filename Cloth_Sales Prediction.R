# Analysizing the business problem
# Target Variable = Sales
# Independent Variables = Other Factors like Income,Adv,price,age

# Importing the dataset

##Exploring and preparing the data
str(Company)

set.seed(123)

Sales = ifelse(Company$Sales<9, "No", "Yes")  # if greater than 8 then high sales else Low
Company1 = data.frame(Company[2:11], Sales)
str(Company1)

# look at the class variable
table(Company1$Sales)
View(Company1)

# Shuffle the data
Company_rand <- Company1[order(runif(400)),]
str(Company_rand)

Company_rand$Sales <- as.factor(Company_rand$Sales)
View(Company_rand)
str(Company_rand)

# split the data frames
train <- Company_rand[1:300, ]
test  <- Company_rand[301:400, ]

# check the proportion of class variable
prop.table(table(Company_rand$Sales))
prop.table(table(train$Sales))
prop.table(table(train$Sales))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

model <- C5.0(train[, -11], train$Sales)

windows()
plot(model) 

# Display detailed information about the tree
summary(model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(model, test)
test_acc <- mean(test$Sales == test_res)
test_acc #  0.71

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(test$Sales, test_res, dnn = c('actual', 'predicted'))

# On Training Dataset
train_res <- predict(model, train)
train_acc <- mean(train$Sales == train_res)
train_acc # 0.9533333

table(train$Sales, train_res)


## Pruning Techinque


library(rpart)
model <- rpart(train$Sales ~ ., data = train,
               control = rpart.control(cp = 0, maxdepth = 3))

# Plot Decision Tree
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = -3)

# Measure the RMSE on Test data
test_pred <- predict(model, newdata = test, type = "vector")

# For Categorical data we can use accuracy,recall,precision or senstivity for the measures

confusion_test <- table(x = test$Sales, y = test_pred)

Accuracy_test <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy_test # 0.76 

# Prune the Decision Tree

# Grow the full tree
fullmodel <- rpart(train$Sales ~ ., data = train,
                   control = rpart.control(cp = 0))

rpart.plot(fullmodel, box.palette = "auto", digits = -3)

# Examine the complexity plot
# Tunning parameter check the value of cp which is giving us minimum cross validation error (xerror)
printcp(fullmodel)   
plotcp(model)

mincp <- model$cptable[which.min(model$cptable[, "xerror"]), "CP"]

# Prune the model based on the optimal cp value
model_pruned_1 <- prune(fullmodel, cp = mincp)
rpart.plot(model_pruned_1, box.palette = "auto", digits = -3)

model_pruned_2 <- prune(fullmodel, cp = 0.02)
rpart.plot(model_pruned_2, box.palette = "auto", digits = -3)

# Measure the Accuracy using Full tree
test_pred_fultree <- predict(fullmodel, newdata = test, type = "vector")
confusion_ftest <- table(x = test$Sales, y = test_pred_fultree)

accuracy_f <- sum(diag(confusion_ftest))/sum(confusion_test)
accuracy_f # 0.73

# Measure the Accuracy using Prune tree
test_pred_prune1 <- predict(model_pruned_1, newdata =  test, type = "vector")

confusion_prune1 <- table(x = test$Sales, y = test_pred_prune1)

accuracy_prune <- sum(diag(confusion_prune1))/sum(confusion_prune1)
accuracy_prune # 0.73

# Measure the Accuracy using Prune tree

test_pred_prune2 <- predict(model_pruned_2, newdata =  test, type = "vector")

confusion_prune2 <- table(x = test$Sales, y = test_pred_prune2)

accuracy_prune2 <- sum(diag(confusion_prune2))/sum(confusion_prune2)
accuracy_prune2

# Prediction for trained data result
train_pred_fultree <- predict(fullmodel, train, type = 'vector')

confusion_tf <- table(x = train$Sales, y = train_pred_fultree)
accuracy_tf <- sum(diag(confusion_tf))/sum(confusion_tf)
accuracy_tf # 0.90


# Prediction for trained data result

train_pred_prune <- predict(model_pruned_1, train, type = 'vector')

confusion_tprune2 <- table(x = train$Sales, y = train_pred_prune)

accuracy_tprune2 <- sum(diag(confusion_tprune2))/sum(confusion_tprune2)
accuracy_tprune2 # 0.90
