# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:41:49 2020

@author: Ozan
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("../DataFile/tennis.csv")

tem = data.iloc[:,1:2]
humidity = data.iloc[:,2:3]

weather = data.iloc[:,0:1].values
le = LabelEncoder()
weather[:,0] = le.fit_transform(weather[:,0])
ohe = OneHotEncoder(categories='auto')
weather=ohe.fit_transform(weather).toarray()

wind = data.iloc[:,3:4].values
le = LabelEncoder()
wind[:,0] = le.fit_transform(wind[:,0])
ohe = OneHotEncoder(categories='auto')
wind=ohe.fit_transform(wind).toarray()

play = data.iloc[:,4:].values
le = LabelEncoder()
play[:,0] = le.fit_transform(play[:,0])
ohe = OneHotEncoder(categories='auto')
play=ohe.fit_transform(play).toarray()


sonuc1 = pd.DataFrame(data = weather, index = range(14), columns=['overcast','rainy','sunny'] )

sonuc2 = pd.DataFrame(data = wind[:,:1], index = range(14), columns=['false'] )

sonuc3 = pd.DataFrame(data = play[:,1:2], index = range(14), columns=['Yes'] )

concat1=pd.concat([sonuc1,tem],axis=1)
concat2=pd.concat([concat1,sonuc2],axis=1)
concat3=pd.concat([concat2,sonuc3],axis=1)


x_train, x_test,y_train,y_test = train_test_split(concat3,humidity,test_size=0.33, random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)

pred=lr.predict(x_test)

X_l = concat3.iloc[:,[0,1,2,3,4,5]].values
r = sm.OLS(endog=humidity,exog=X_l).fit()
print(r.summary())

X_l = concat3.iloc[:,[0,1,2,5]].values
r = sm.OLS(endog=humidity,exog=X_l).fit()
print(r.summary())


X_l = concat3.iloc[:,[0,1,2]].values
r = sm.OLS(endog=humidity,exog=X_l).fit()
print(r.summary())



x_train, x_test,y_train,y_test = train_test_split(sonuc1,humidity,test_size=0.33, random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)

pred=lr.predict(x_test)






