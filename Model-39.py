# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:21:07 2020

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

#Splitting data

train_df,test_df=train_test_split(df,test_size=0.4)

#Bootstrapping training set

training_set=[]

for j in range(100):
    id_pick = np.random.choice(train_df.shape[0], size=(train_df.shape[0]))
    boostrapped_training_set=train_df.iloc[id_pick,:]
    training_set.append(boostrapped_training_set)

a=training_set[0]
i=1
while i<=99:
    a=pd.concat([a,training_set[i]])
    i=i+1

training_set=a

#Bootstrapping testing set

testing_set=[]

for j in range(1000):
    id_pick = np.random.choice(test_df.shape[0], size=(test_df.shape[0]))
    boostrapped_testing_set=test_df.iloc[id_pick,:]
    testing_set.append(boostrapped_testing_set)

a=testing_set[0]
i=1
while i<=999:
    a=pd.concat([a,testing_set[i]])
    i=i+1

testing_set=a

#Building models

test_data_accuracy=[]
predictions=[]
matrix=[]

mean_missclass=[]
i=1
while i<=39:
    model=LogisticRegression().fit(training_set.iloc[:,:i],training_set.iloc[:,39])
    score=model.score(testing_set.iloc[:,:i],testing_set.iloc[:,39])
    predictions.append(model.predict(testing_set.iloc[:,:i]))
    matrix.append(metrics.confusion_matrix(testing_set.iloc[:,39],model.predict(testing_set.iloc[:,:i])))
    test_data_accuracy.append(score) 
    i=i+1


mean_accuracy=[]
mean_accuracy=np.mean(test_data_accuracy)
    

sensitivity=[]
specificty=[]
missclassifications=[]
for i in range(39):
    missclassifications.append(matrix[i][0][1]+matrix[i][1][0])
    sensitivity.append(matrix[i][1][1]/(matrix[i][1][0]+matrix[i][1][1]))
    specificty.append(matrix[i][0][0]/(matrix[i][0][1]+matrix[i][0][0]))


print(max(test_data_accuracy))
print(test_data_accuracy.index(max(test_data_accuracy)))        
print(min(test_data_accuracy))
print(test_data_accuracy.index(min(test_data_accuracy)))

print(max(missclassifications))
print(missclassifications.index(max(missclassifications)))        
print(min(missclassifications))
print(missclassifications.index(min(missclassifications))) 


print(max(sensitivity))
print(sensitivity.index(max(sensitivity)))        
print(max(specificty))
print(specificty.index(max(specificty)))


fig, ax = plt.subplots()
ax.plot(np.arange(1,40,1),sensitivity, color="C0",label="Sensitivity")
plt.xlabel('Models with number of independent variables')
plt.ylabel('Sensitivity')
ax.annotate((round(max(sensitivity),2),sensitivity.index(max(sensitivity))+1),xy =(sensitivity.index(max(sensitivity)),max(sensitivity)),arrowprops= dict(facecolor ='green',shrink = 0.05))
legend = ax.legend(loc='upper center')
plt.show()

fig, ax = plt.subplots()
ax.plot(np.arange(1,40,1),specificty, color="C0",label="Specificity")
plt.xlabel('Models with number of independent variables')
plt.ylabel('Specificity')
ax.annotate((round(max(specificty),3),specificty.index(max(specificty))+1),xy =(specificty.index(max(specificty)),max(specificty)),arrowprops= dict(facecolor ='green',shrink = 0.05))
legend = ax.legend(loc='upper center')
plt.show()

fig, ax = plt.subplots()
ax.plot(np.arange(1,40,1),missclassifications, color="C0",label="Missclassifications")
plt.xlabel('Models with number of independent variables')
plt.ylabel('Number of missclassifications per 20000 data points')
ax.annotate((min(missclassifications),missclassifications.index(min(missclassifications))+1),xy =(missclassifications.index(min(missclassifications)),min(missclassifications)),arrowprops= dict(facecolor ='green',shrink = 0.05))
legend = ax.legend(loc='lower left')
plt.show()

fig, ax = plt.subplots()
ax.plot(np.arange(1,40,1),test_data_accuracy, color="C0",label="Accuracy")
plt.xlabel('Models with number of independent variables')
plt.ylabel('Accuracy')
ax.annotate((round(max(test_data_accuracy),3),test_data_accuracy.index(max(test_data_accuracy))+1),xy =(test_data_accuracy.index(max(test_data_accuracy)),max(test_data_accuracy)),arrowprops= dict(facecolor ='green',shrink = 0.05))
legend = ax.legend(loc='upper left')
plt.show()
    

import pickle
variables={"sensitivity":sensitivity,"specificty":specificty,"training_set":training_set,"testing_set":testing_set,"test_data_accuracy":test_data_accuracy,"missclassifications":missclassifications}
with open("variables.pickle","wb") as f:
    pickle.dump(variables, f)


