# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:43:33 2017

@author: victor
"""

import pandas as pd
import numpy as np

data = pd.read_csv('train_cleaned_v4.csv')
data_test = pd.read_csv('test_cleaned_v4.csv')
id_test = data_test['id']
y = data.loc[:,'Y']
X = data.loc[:,'F6':]
np.random.seed(20)
#X = preprocessing.scale(X)
X = np.array(X)
y = np.array(y)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X,y,test_size = 0.2)
test = data_test.loc[:,'F6':]

knn = KNeighborsClassifier()
knn_param_grid = {'n_neighbors':range(200,1001,200)}
kfold = StratifiedKFold(n_splits=2)
clf = GridSearchCV(knn,param_grid = knn_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 1, verbose = 100)

clf.fit(X_all_train,y_all_train)

print(clf.best_params_)
print(clf.best_score_)