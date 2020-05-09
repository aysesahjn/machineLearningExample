# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:01:45 2020

@author: Ozan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

data = pd.read_csv("../DataFile/salary.csv")

x=data.iloc[:,1:2]
y=data.iloc[:,-1:]
xValue=x.values
yValue=y.values

#Karar ağacı
tr= DecisionTreeRegressor(random_state=0)
tr.fit(xValue,yValue)

plt.scatter(xValue,yValue)
plt.plot(xValue,tr.predict(xValue), color='red')
plt.show()

