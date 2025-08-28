#Import required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#import dataset
st_data = pd.read_csv('50_startups.csv')

print(st_data.head())


X = st_data.iloc[:, :-1] #store first columns in x variable
y = st_data.iloc[:, 4] #store last column in y variable

#print(X, y)

#Convert state colum into categorical columns
states = pd.get_dummies(X['State'], drop_first=True, dtype=int)

#Drop State Column
X=X.drop('State', axis=1)

#Concat
X=pd.concat([X, states], axis=1)


#import  Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#import linear regression model
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#Predicting test set result
y_pred = linear_model.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)

print("R2 Score", score)


