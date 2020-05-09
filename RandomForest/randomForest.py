# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:19:50 2020

@author: Ozan
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = pd.read_csv("../DataFile/salary.csv")

x=data.iloc[:,1:2]
y=data.iloc[:,-1:]
xValue=x.values
yValue=y.values

#Karar ağacı
rfr= RandomForestRegressor(n_estimators=100,random_state=0)
rfr.fit(xValue,yValue)

plt.scatter(xValue,yValue)
plt.plot(xValue,rfr.predict(xValue), color='red')
plt.show()

