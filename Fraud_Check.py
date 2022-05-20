import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer


Fraud = pd.read_csv("C:/Users/personal/Desktop/Fraud_check.csv")

Fraud.columns

Fraud.columns = "UG","MS","TI","CP","WE","Urban"

Fraud.info()

# Creating Dummy Variables
lb = LabelEncoder()
Fraud["UG"] = lb.fit_transform(Fraud["UG"])
Fraud["MS"] = lb.fit_transform(Fraud["MS"])
Fraud["Urban"] = lb.fit_transform(Fraud["Urban"])
Fraud["TI"] = np.digitize(Fraud["TI"], [30000, 60000, 90000 ])
Fraud

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
Fraud_n = norm_func(Fraud.iloc[:, 3:5])
Fraud.info()
Fraud_n.describe()

Fraud_n

Fraud_f = pd.concat([Fraud,Fraud_n], axis = 1)

Fraud_f.columns = "UG","MS","TI","CP","WE","Urban","CPN","WEN"

Fraud_f = Fraud_f.drop(['CP','WE'], axis = 1)


X = np.array([[Fraud_f.UG,Fraud_f.MS,Fraud_f.Urban,Fraud_f.CPN,Fraud_f.WEN]]) # Predictors 
Y = np.array(Fraud_f['TI']) # Target 

X.shape
Y.shape

X = X.reshape(X.shape[1 :])
X.shape

X = X.transpose()
Y = Y.transpose()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier

help(RandomForestClassifier)
rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

rf.fit(X_train, Y_train) # Fitting RandomForestClassifier model from sklearn.ensemble  

pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score

pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions'])

print(accuracy_score(Y_test, pred)) #0.744

# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2

# train accuracy 
train_acc2 = np.mean(rf.predict(X_train)==Y_train)
train_acc2 # 0.99






