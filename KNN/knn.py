# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:33:50 2020

@author: Ozan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("../DataFile/data.csv")

veri=data.iloc[:,1:4]
cinsiyet=data.iloc[:,-1:]

#Eğitim test ayır
x_train, x_test, y_train, y_test = train_test_split(veri,cinsiyet,test_size=0.33,random_state=0)

#Verilerin ölçeklenmesi
sc=StandardScaler()
veri_s=sc.fit_transform(x_train)
cinsiyet_s=sc.transform(x_test)

#KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train,y_train)
pred=knn.predict(x_test)

#Tahminin ne kadar doğru sınıflandırıldığını Confusing Matrix ile baktık
cm = confusion_matrix(y_test,pred)
print(cm)

