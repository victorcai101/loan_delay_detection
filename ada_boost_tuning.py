# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:52:58 2017

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Adaboost
#DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(random_state=7)

ada_param_grid = {
              "n_estimators" :range(100,701,100),#200
              "learning_rate":  [0.001, 0.01, 0.1, 0.3, 0.75,1.5]}#0.1

from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=5)
gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 1, verbose = 100)

gsadaDTC.fit(X_all_train,y_all_train)

print(gsadaDTC.best_params_)
print(gsadaDTC.best_score_)