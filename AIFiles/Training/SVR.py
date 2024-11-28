#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:27:14 2022

@author: patrikchrenko

Using lecture 5 exercises
"""

#library imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.svm import SVR

#load dataset
path = "../Data/Laundried/"  
filename_read = os.path.join(path, "nullCleaned.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

droprows = df.index[(np.abs(df['SalePrice'] - df['SalePrice'].mean()) >= (2*df['SalePrice'].std()))]
df.drop(droprows,axis=0,inplace=True)

# Strip non-numerics
df = df.select_dtypes(include=['int', 'float'])

result = []
for x in df.columns:
    if x != 'SalePrice':
        result.append(x)
   
X = df[result].values
y = df['SalePrice'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

#c Create LinReg
linReg = LinearRegression().fit(X_train, y_train)

# Predict values
y_pred = linReg.predict(X_test)

# Compare actual to test data
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

# Evaluate accuracy
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 
print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))
print('Correlation: ', stats.pearsonr(y_test,y_pred))


#plot the outputs
#set figure size
plt.rc('figure', figsize=(5, 5))

#plot line down the middle
x = np.linspace(350000,0,10)
plt.plot(x, x, '-r')

#plot the points, prediction versus actual
plt.scatter(y_test, y_pred, color='black')

plt.xticks(())
plt.yticks(())

plt.show()

#and plot the values to emphasise the noise
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression(y_pred,y_test,sort=True)  
