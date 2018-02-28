# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:17:33 2017

@author: victor
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer

data = pd.read_csv('train_cleaned_v10.csv')
data_test = pd.read_csv('test_cleaned_v10.csv')
id_test = data_test['id']
y = data.loc[:,'Y']
X = data.loc[:,'F6':]

np.random.seed(0)
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X,y,test_size = 0.2)

test = data_test.loc[:,'F6':]
n_fold = 5

from sklearn.model_selection import GridSearchCV, StratifiedKFold
n_fold = StratifiedKFold(n_splits=3)
parameters = {"min_samples_leaf":[14],
              "n_estimators":[500]}
#"min_samples_leaf":25
#"min_samples_split":[100]
auc_scorer = make_scorer(roc_auc_score)
RFC = RandomForestClassifier(random_state=50,max_depth=10,n_estimators=300, 
                             max_features = "auto", min_samples_leaf=25, 
                             min_samples_split=100,min_weight_fraction_leaf=0.0005)
#clf = GridSearchCV(RFC,parameters,cv=5,scoring=auc_scorer)
clf = GridSearchCV(RFC,parameters,cv=n_fold,scoring='roc_auc', n_jobs = 1,verbose=100)
clf.fit(X_all_train,y_all_train)
print(clf.best_params_)
print(clf.best_score_)
#RFC.fit(X_all_train,y_all_train)
#y_hat = RFC.predict_proba(X_all_test)[:,1]
#score = roc_auc_score(y_all_test,y_hat)
#y_hat = clf.predict_proba(test)[:,1]


'''
to_save = pd.DataFrame(id_test)
to_save['Y'] = y_hat
to_save.to_csv('rf__min_weight_fraction_leaf0.001.csv',index=False)
'''