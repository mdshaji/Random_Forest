# random forest evaluation for Taxable income

str(Fraud_check)

attach(Fraud_check)

View(Taxable.Income)

# converting Taxable income into categorical

for (i in 1:nrow(Fraud_check))
  
  if(Fraud_check$Taxable.Income[i] >= 30000){
    Fraud_check$Taxable.Income[i] <- "Good"
  } else{
    Fraud_check$Taxable.Income[i] <- "risky"
  }

View(Fraud_check$Taxable.Income)

attach(Fraud_check)

table(Fraud_check$Taxable.Income)

Fraud_check$Taxable.Income <- as.factor(Fraud_check$Taxable.Income)

round(prop.table(table(Fraud_check$Taxable.Income))*100, 1)


#Create a function to normalize the data
normFunc <- function(x){(x-mean(x, na.rm = T))/sd(x, na.rm = T)}

Fraud_check[4:5] <- apply(Fraud_check[4:5], 2, normFunc)

View(Fraud_check)

Fraud_rand <- Fraud_check[order(runif(600)), ]
View(Fraud_rand)

# create training and test data
Fraud_train <- Fraud_rand[1:450, ]
View(Fraud_train)
Fraud_test <- Fraud_rand[450:600, ]

# create labels for training and test data
Fraud_train_labels <- Fraud_rand[1:450, 3]
Fraud_test_labels <- Fraud_rand[450:600, 3]

str(Fraud_check)
Fraud_check$Taxable.Income = as.factor(Fraud_check$Taxable.Income)

# check for proportions

prop.table(table(Fraud_rand$Taxable.Income))
prop.table(table(Fraud_train$Taxable.Income))
prop.table(table(Fraud_test$Taxable.Income))

# Build the model using C5.0

library(C50)

Fraud_model <- C5.0(Fraud_train[, -3], Fraud_train$Taxable.Income)

plot(Fraud_model)

# Display detailed information about the tree
summary(Fraud_model) # Highest Information gain is from price so this is the root node

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(Fraud_model, Fraud_test)
test_acc <- mean(Fraud_test$Taxable.Income == test_res)
test_acc #0.761 is the test accuracy

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(Fraud_test$Taxable.Income, test_res, dnn = c('actual TI', 'predicted TI'))

# On Training Dataset
train_res <- predict(Fraud_model, Fraud_train)
train_acc <- mean(Fraud_train$Taxable.Income == train_res)
train_acc # 0.80

table(Fraud_train$Taxable.Income, train_res)

# Building a random forest model on training data 
install.packages("randomForest")
library(randomForest)

?randomForest
Fraud_forest <- randomForest(Fraud_train_labels  ~ ., data = Fraud_train, importance = TRUE)
plot(Fraud_forest)

# Test Data Accuracy
test_acc <- mean(Fraud_test$Taxable.Income == predict(Fraud_forest, newdata=Fraud_test))
test_acc

# Train Data Accuracy
train_acc <- mean(Fraud_train$Taxable.Income == predict(Fraud_forest, data=Fraud_train))
train_acc 







