# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:41:19 2020

@author: Ozan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("../DataFile/salary.csv")

x=data.iloc[:,1:2]
y=data.iloc[:,-1:]
xValue=x.values
yValue=y.values

#Karar ağacı
tr= DecisionTreeRegressor(random_state=0)
tr.fit(xValue,yValue)
#Rasal ağaçlar
rfr= RandomForestRegressor(n_estimators=100,random_state=0)
rfr.fit(xValue,yValue)

print('R Score')
print(r2_score(yValue,tr.predict(xValue)))
print(r2_score(yValue,rfr.predict(xValue)))
#Buraya diğer regressor tahminlerini de koyup kıyaslama yapabilirsin. Sadece R2 değerine bakmak yanıltır.
#Algoritmaların yapısını da bilmek gerekiyor. Örneğin Decision Tree score'u en yüksek değer olan 1 çıkıyor
#Bu yanıltıcı bir değerdir. Decision Tree değerleri belli bir alana kaşılık görür ve o alanda her değer 
#için aynı tahmini verir.