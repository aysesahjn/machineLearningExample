# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:54:14 2020

@author: Ozan
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data = pd.read_csv("dataFile\salary.csv")

x=data.iloc[:,1:2]
y=data.iloc[:,-1:]
xValue=x.values
yValue=y.values

#Linear Regression
lr=LinearRegression()
lr.fit(xValue,yValue)

plt.scatter(xValue,yValue)
plt.plot(xValue, lr.predict(xValue), color='red')
plt.show()

#Polynomial Regression
pr=PolynomialFeatures(degree=2)
x_poly=pr.fit_transform(xValue)
lr2=LinearRegression()
lr2.fit(x_poly,y)

plt.scatter(xValue,yValue)
plt.plot(xValue, lr2.predict(x_poly), color='red')
plt.show()

