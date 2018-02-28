# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:55:39 2017

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
from sklearn.model_selection import train_test_split
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X,y,test_size = 0.2)
test = data_test.loc[:,'F6':]
from sklearn.ensemble import ExtraTreesClassifier
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [10],
              "min_samples_split": [15],
              "min_samples_leaf": [10],
              "bootstrap": [False],
              "n_estimators" :[200],
              "criterion": ["gini"],
              }

from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=5)
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 1, verbose = 100)

gsExtC.fit(X_all_train,y_all_train)
print(gsExtC.best_params_)
ExtC_best = gsExtC.best_estimator_

# Best score
print(gsExtC.best_score_)