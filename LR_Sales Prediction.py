# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:30:02 2019

@author: dongfrank
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font",size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split 
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True) 

data=pd.read_csv('F:/bank.csv',delimiter=';')
data=data.dropna()
#print(data.shape)
#print(list(data.columns))
#
#data.head() 

data['education'].unique()  #show education's objects 
data['education']=np.where(data['education']=='basic.4y','basic',data['education'])
data['education']=np.where(data['education']=='basic.6y','basic',data['education'])
data['education']=np.where(data['education']=='basic.9y','basic',data['education'])

data.info() #check type of data 
data.loc[data['y']=='yes','y']=1  
data.loc[data['y']=='no','y']=0
data['y'].value_counts()  #count the number 0 or 1 
sns.countplot(x='y',data=data,palette='hls') 

data.groupby('y').mean()
data.groupby('education').mean() 

#Create virtual variance 
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','pout]

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list=pd.get_dummies(data[var],prefix=var) #one-hot encoding
    data1=data.join(cat_list) 
    data=data1
    
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars] 

data_final=data[to_keep]
data_final.columns.values 

data_final_vars=data_final.columns.values.tolist()
y=['y']
x=[i for in in data_final_vars if i not in y] 

#RFE Recursive Feature Elimniation 
from sklearn import datasets 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression 
logreg=LogisticRegression() 
rfe=RFE(estimator=logreg,n_features_to_select=18) 
rfe=rfe.fit(data_final[X],data_final[y])
print(rfe.support_) 
print(rfe.ranking_)

from itertools import compress
cols=list(compress(X,rfe.support_)) 

#Build the model 
import statsmodels.api as sm

X=data_final[cols]
y=data_final['y'] 

logit_model=sm.Logit(y,X)
logit_model.raise_on_perfect_prediction=False 
result=logit_model.fit() 
print(result.summary().as_text) 

#LR model
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0) 
logreg=LogisticRegression() 
logreg.fit(X_train,y_train) 
y_pred=logreg.predict(X_test) 
y_pred=logreg.predict(X_test) 
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test))) 

#Cross-Validation
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold=model_selection.KFold(in_splits=10,random_state=7) 
modelCV=LogisticRegression() 
scoring='accuracy'
results=model_selection.cross_val_score(modelCV,X_train,y_train,cv=kfold,scoring=scoring) 
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred)) 

#ROC Curve 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 

logit_roc_auc=roc_auc_score(y_test,logreg.predict(X_test)) 
fpr,tpr,thresholds=roc_curve(y_test,logreg.predict_proba(X_test)[:,1]) 
plt.figure() 
plt.plot(fpr,tpr,label='Logistic Regression(area=%0.2f)' %logit_roc_auc) 
plt.plot([0,1],[0,1],'r--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()



























