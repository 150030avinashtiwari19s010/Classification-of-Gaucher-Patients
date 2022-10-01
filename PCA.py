# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:51:03 2020

@author: AVINASH
"""

#Loading packages

import scipy.io
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#Loading data

data=scipy.io.loadmat('Gaucherdata_FIRST_EDIT.mat')

#Editing data

label=data['label']
moverzNEW=data['moverzNEW']
moverzNEW=pd.DataFrame(moverzNEW)
Xfinal=data['Xfinal']
Xfinal=pd.DataFrame(Xfinal)
result=[0]*40
i=0
while i<=39:
    r=str(label[i,])
    r=r[9:16]
    result[i]=r
    i=i+1    
for i in range(len(result)):
    if result[i]=='control':
        result[i]=0
    else:
        result[i]=1
result=pd.DataFrame(result)
df=pd.concat([Xfinal,result],axis=1)

label1=data['label']
moverzNEW1=data['moverz']
moverzNEW1=pd.DataFrame(moverzNEW1)
Xfinal1=data['X']
Xfinal1=pd.DataFrame(Xfinal1)
result1=[0]*40
i=0
while i<=39:
    r=str(label1[i,])
    r=r[9:16]
    result1[i]=r
    i=i+1    
for i in range(len(result1)):
    if result1[i]=='control':
        result1[i]=0
    else:
        result1[i]=1
result1=pd.DataFrame(result1)
df1=pd.concat([Xfinal1,result1],axis=1)

E=pd.DataFrame(data['Ediag'])
A=pd.DataFrame(E.cumsum()/E.sum()*100)


fig, ax = plt.subplots()
ax.plot(moverzNEW.iloc[0,:],E.iloc[551:,0], color="C0",label="Eigen Values")
plt.xlabel('m/z values')
plt.ylabel('Eigen values')
legend = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.plot(moverzNEW.iloc[0,:],A.iloc[551:,0],'r+',label=" % Variance")
ax2.yaxis.set_major_formatter(PercentFormatter())
plt.ylabel("Percentage of variance explained")
legend = ax2.legend(loc='upper center')
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")

plt.show()



fig, ax = plt.subplots()
ax.plot(moverzNEW1.iloc[0,:],E.iloc[:,0], color="C0",label="Eigen Values")
plt.xlabel('m/z values')
plt.ylabel('Eigen values')
legend = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.plot(moverzNEW1.iloc[0,:],A.iloc[:,0],'r+',label=" % Variance")
ax2.yaxis.set_major_formatter(PercentFormatter())
plt.ylabel("Percentage of variance explained")
legend = ax2.legend(loc='upper center')
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")

plt.show()

