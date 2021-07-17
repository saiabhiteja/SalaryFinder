# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 00:35:10 2021

@author: saiabhiteja
"""

#importing libraries
import pandas as pd
import pickle
df = pd.read_csv("Salary_Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)
#saving pkl file using pickle module
pickle.dump(lr,open('model.pkl','wb'))
