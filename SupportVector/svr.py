# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:01:45 2020

@author: Ozan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data = pd.read_csv("dataFile\salary.csv")

x=data.iloc[:,1:2]
y=data.iloc[:,-1:]
xValue=x.values
yValue=y.values

#SVR için değerlerin önce Scale edilmesi gerekiyor
sc1=StandardScaler()
x_s=sc1.fit_transform(xValue)
sc2=StandardScaler()
y_s=sc1.fit_transform(yValue)

#SVR
svr=SVR(kernel='rbf')
svr.fit(x_s,y_s)

plt.scatter(x_s,y_s)
plt.plot(x_s,svr.predict(x_s), color='red')
plt.show()

