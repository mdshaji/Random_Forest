# Analysizing the business problem
# Target Variable = Class Variable - Predicting the Diabetes
# Independent Variables = Other Factors 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
Diabetes = pd.read_csv("C:/Users/personal/Desktop/Diabetes.csv")
Diabetes.columns

# Renaming the column names
Diabetes.columns = "NP","PGC","BP","SFT","SI","BMI","DPF","Age","CV"

# Creation of a Dummy Variable for output
lb = LabelEncoder()
Diabetes["CV"] = lb.fit_transform(Diabetes["CV"])

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
Diabetes_n = norm_func(Diabetes.iloc[:, 0:9])
Diabetes.info()
Diabetes_n.describe()

Diabetes_n
X = np.array(Diabetes_n.iloc[:,0:8]) # Predictors 
Y = np.array(Diabetes_n['CV']) # Target 

# Model Building
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier

help(RandomForestClassifier)
rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

rf.fit(X_train, Y_train) # Fitting RandomForestClassifier model from sklearn.ensemble  

pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score

pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions'])

print(accuracy_score(Y_test, pred)) #0.77

# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2 # 0.77

# train accuracy 
train_acc2 = np.mean(rf.predict(X_train)==Y_train)
train_acc2 # 0.99

# Conclusion
# From the train and test accuracy we can say that still the model is overfit


