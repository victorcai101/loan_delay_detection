# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:33:09 2017

@author: victor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score

data = pd.read_csv('train_cleaned_v1.csv')
y = data.loc[:,'Y']
X = data.loc[:,'F1':]
data_test = pd.read_csv('test_cleaned_v1.csv')
id_test = data_test['id']
test = data_test.loc[:,'F1':]
test =  preprocessing.scale(test)
np.random.seed(0)
X = preprocessing.scale(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X,y,test_size = 0.2)

RFC = RandomForestClassifier(max_depth=11, n_estimators=195,random_state=0)
rfecv = RFECV(estimator=RFC, step=1, cv=StratifiedKFold(5),scoring='roc_auc')
rfecv.fit(X_all_train, y_all_train)

AUC = []
mdl_list=[]


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

X_transform_train = rfecv.transform(X_all_train)
X_transform_test = rfecv.transform(X_all_test)
n_fold = 5
kf = KFold(n_splits = n_fold)
for train_index, test_index in kf.split(X_transform_train):
    X_ftrain, X_ftest = X_transform_train[train_index], X_transform_train[test_index]
    y_ftrain, y_ftest = y_all_train[train_index], y_all_train[test_index]

    mdl = RandomForestClassifier(max_depth=11, n_estimators=195,random_state=0)
    mdl.fit(X_ftrain,y_ftrain)
    mdl_list.append(mdl)
    
    #y_test_hat = mdl_list[1].predict_proba(X_all_test[:,col_name])[:,1]
    #score = roc_auc_score(y_all_test,y_test_hat)

real_test_transform = rfecv.transform(test)

y_hat = np.zeros(50000)
for mdl in mdl_list: 
    y_hat = y_hat+mdl.predict_proba(real_test_transform)[:,1]/len(mdl_list)
to_save = pd.DataFrame(id_test)
to_save['Y'] = y_hat
to_save.to_csv('recursive_selection.csv',index=False)
'''
score = roc_auc_score(y_all_test,y_hat)
print(score)
'''