# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:45:16 2017

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
kfold = StratifiedKFold(n_splits=5)
from sklearn.neural_network import MLPClassifier
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X,y,test_size = 0.2)
test = data_test.loc[:,'F6':]

parameters={
'learning_rate': ["invscaling"],
'hidden_layer_sizes': [(100,3)],
'alpha': [0.01],
'activation': ["tanh"]
}

clf = GridSearchCV(estimator=MLPClassifier(),param_grid=parameters,n_jobs=1,scoring='roc_auc',verbose=100,cv=kfold)
clf.fit(X_all_train,y_all_train)

print(clf.best_params_)
print(clf.best_score_)