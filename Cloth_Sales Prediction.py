# Analysizing the business problem
# Target Variable = Sales
# Independent Variables = Other Factors like Income,Adv,price,age

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer


Company = pd.read_csv("C:/Users/personal/Desktop/Company_data.csv")

Company.isnull().sum()
Company.dropna()
Company.columns

Company.info()

# Creating Dummy Variables
lb = LabelEncoder()
Company["ShelveLoc"] = lb.fit_transform(Company["ShelveLoc"])
Company["Urban"] = lb.fit_transform(Company["Urban"])
Company["US"] = lb.fit_transform(Company["US"])

# Discretizing sales 
Company["Sales"] = np.sort(Company["Sales"])
Company["Sales"]

Company["Sales"] = np.digitize(Company["Sales"], [0, 3, 6, 9, 12, 17])

Company['Sales'].unique()
Company['Sales'].value_counts()
colnames = list(Company.columns)

predictors = colnames[1:11]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(Company, test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])
np.mean(preds == test[target])
confusion_matrix = pd.crosstab(preds, test[target])
confusion_matrix

accuracy_test = 28/80 = 


 # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target])
###################################################################################

# prune the above data

Company = pd.read_csv("C:/Users/personal/Desktop/Company_data.csv")

#converting into binary
lb = LabelEncoder()
Company["ShelveLoc"] = lb.fit_transform(Company["ShelveLoc"])
Company["Urban"] = lb.fit_transform(Company["Urban"])
Company["US"] = lb.fit_transform(Company["US"])

# discretizing sales 
Company["Sales"] = np.sort(Company["Sales"])
Company["Sales"]

Company["Sales"] = np.digitize(Company["Sales"], [0, 3, 6, 9, 12, 17])


Company['Sales'].unique()
Company['Sales'].value_counts()
colnames = list(Company.columns)

x = colnames[1:11]
y = colnames[0]
Company.info()
Company.columns

x = np.array([[Company.CompPrice, Company.Income, Company.Advertising, Company.Population, Company.Price, Company.ShelveLoc, Company.Age, Company.Education, Company.Urban, Company.US]])
y = np.array([Company.Sales])

x.shape
y.shape

x = x.reshape(x.shape[1 :])
x.shape

x = x.transpose()
y = y.transpose()
# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score
# accuracy on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred) #0.88

# accuracy on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred) # 0.82


# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# accuracy on test dataset
mean_squared_error(y_test, test_pred2) #1.72
r2_score(y_test, test_pred2)

# accuracy on train dataset
mean_squared_error(y_train, train_pred2) #0.02
r2_score(y_train, train_pred2)#0.98

###########
## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# accuracy on test dataset
mean_squared_error(y_test, test_pred3) #1.543
r2_score(y_test, test_pred3) 

# accuracy on train dataset
mean_squared_error(y_train, train_pred3) #0.20
r2_score(y_train, train_pred3) # 0.78



