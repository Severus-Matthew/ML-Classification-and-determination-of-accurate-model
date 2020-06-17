#CLASSIFICATION OF DATA
#importing libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

#creating and encoding dataset

dataset= pd.read_csv("IRIS.csv")
dataset["species"]=dataset["species"].astype('category')
dataset["encoded"]= dataset["species"].cat.codes
print(dataset.head())

#creating test train model

X_data = dataset.iloc[:,:4]
Y_data = dataset.iloc[:,5]
X_train , X_test , Y_train , Y_test = train_test_split(X_data , Y_data , test_size = 0.1)

#LOGISTIC REGRESSION
#importing libraries

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#Fitting data &confusion matrix

LogReg = LogisticRegression()
LogReg.fit(X_train , Y_train)
Y_pred = LogReg.predict(X_test)
confusion_matrix(Y_test , Y_pred)

#accuracy

LoAc= accuracy_score(Y_test , Y_pred)

#classification report

print(classification_report(Y_test , Y_pred))

#K NEIGHBOUR MODEL
#Import libraries

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#Fitting data & Confusion matrix

knn= KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred= knn.predict(X_test)
confusion_matrix(Y_test, Y_pred)

#accuracy 

KNNAc= accuracy_score(Y_test , Y_pred)

#Classification report

print(classification_report(Y_test , Y_pred))

#DECISION TREE
#Importing libraries

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#Fitting data & Confusion matrix

destr = DecisionTreeClassifier()
destr.fit(X_train , Y_train)
Y_pred = destr.predict(X_test)
confusion_matrix(Y_test, Y_pred)

#accuracy

DTAc= accuracy_score(Y_test , Y_pred)

#classification report

print(classification_report(Y_test , Y_pred))

#RANDOM FOREST
#importing libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#Fitting data & Confusion matrix

retr = RandomForestClassifier()
retr.fit(X_train, Y_train)
Y_pred= retr.predict(X_test)
confusion_matrix(Y_test, Y_pred)

#accuracy

RFAc= accuracy_score(Y_test , Y_pred)

#Classification report

print(classification_report(Y_test , Y_pred))

#ACCURACY COMPARISON

print("accuracy of Logisctic Regression" )
print(LoAc)
print("accuracy of KNeighbour Model")
print(KNNAc)
print("accuracy of Decisison Tree")
print(DTAc)
print("accuracy of Random Forest")
print(RFAc)
m= max(LoAc , DTAc, KNNAc, RFAc)
print("maxium accuracy")
print(m)

