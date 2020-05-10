# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:04:16 2020

@author: Ozan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("../DataFile/data.csv")

veri=data.iloc[:,1:4]
cinsiyet=data.iloc[:,-1:]

#Eğitim test ayır
x_train, x_test, y_train, y_test = train_test_split(veri,cinsiyet,test_size=0.33,random_state=0)

#Verilerin ölçeklenmesi
sc=StandardScaler()
veri_s=sc.fit_transform(x_train)
cinsiyet_s=sc.transform(x_test)

logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

pred=logr.predict(x_test)









