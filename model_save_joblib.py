# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:32:43 2020

@author: Vishwajeet
"""

# ML models can be saved using pickle or joblib.
# COns of Pickle: it dosen't saves test results and any data. also it is less secure.
# joblib is handy for larger dataset

# Saving model using joblib::
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg','pias','pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(url, names = names)
array = df.values
X = array[:,0:8]
Y = array[:,8]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = .33, random_state=21)

model = LogisticRegression()
model.fit(X_train,Y_train)

file_ = "LR_model.pkl"   #.sav is another format
joblib.dump(model, file_)

_model_ = joblib.load(file_)
print(_model_.predict(X_test))
print("score::", _model_.score(X_test, Y_test))
