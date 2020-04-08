# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:33:53 2020

@author: KyoChan
"""


import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.precision',3)
import matplotlib.pyplot as plt
import seaborn as sns
#def MSE
def MSE(y_true, y_pred):
    return np.mean(np.square((y_true -y_pred)))

#Importing the dataset
bike = pd.read_csv('C:\RExercise\ESC\day.csv')
bike

bike.info()
Data = bike[['cnt', 'temp']]
Data
Data.describe()


#Data.hist(bins=50, figsize=(12,5))
#plt.tight_layout()

#sns.pairplot(Data)

data = Data.sample(n=30).reset_index()
data.shape
#일부러 적게 뽑았다? *731개 중 30개

N= data.shape[0];N
test = np.random.choice(np.arange(N),20,replace=False)
train_df = data[~data.index.isin(test)].copy()
test_df = data[data.index.isin(test)].copy()
print(train_df.shape, test_df.shape)

y=np.matrix(data['cnt']).T
X=np.matrix(data['temp']).T
y_train = np.matrix(train_df['cnt']).T
X_train = np.matrix(train_df['temp']).T
y_test = np.matrix(test_df['cnt']).T
X_test = np.matrix(test_df['temp']).T

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
lin2 = LinearRegression(fit_intercept=False)
X_train
MSE_log=np.array(range(1,7));MSE_log
MSE_log_test=np.array(range(1,7))
for i in range(1,7):
    j=i*2
    poly = PolynomialFeatures(degree=j);poly
    #Train MSE
    X_train_poly = poly.fit_transform(X_train);X_train_poly 
    lin2.fit(X_train_poly,y_train)
    y_hat = lin2.predict(X_train_poly)
    MSE_log[i-1]=(np.log10(MSE(y_train,y_hat)))
    #Test MSE
    lin2.coef_.T
    X_test_poly = poly.fit_transform(X_test)
    y_test_hat = np.dot(X_test_poly, lin2.coef_.T)
    MSE_log_test[i-1]=(np.log10(MSE(y_test,y_test_hat)))

plt.plot(np.array(range(1,7))*2,MSE_log,color='r')
plt.plot(np.array(range(1,7))*2,MSE_log_test,color='b')

#add to plot by sth
#find model
#use X_test_poly, y_test, find MSE
#X_test_poly = poly.fit_transform(X_test)
#lin2.fit(X_test)
#add to plot samely   
  



