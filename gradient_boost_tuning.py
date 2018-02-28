# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 18:34:28 2017

@author: victor
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from class_SklearnHelper import SklearnHelper,get_oof
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.cross_validation import KFold
#col_name = [0,1,3,4,5,6,7,8,9,11,13,14,15,16,17,18,19,20,21,22,23,24,26]

data = pd.read_csv('train_cleaned_v4.csv')
data_test = pd.read_csv('test_cleaned_v4.csv')
id_test = data_test['id']
y = data.loc[:,'Y']
X = data.loc[:,'F6':]

np.random.seed(0)
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X,y,test_size = 0.2)

test = data_test.loc[:,'F6':]

ntrain = X_all_train.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)


'''
gbm0 = GradientBoostingClassifier(random_state=10)
param_test1 = {'n_estimators':range(20,121,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
                                                               min_samples_leaf=50,max_depth=8,max_features='sqrt',
                                                               subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=100)
gsearch1.fit(X_all_train,y_all_train)
print(gsearch1.cv_results_)
print(gsearch1.best_params_)

gbm1 = GradientBoostingClassifier(random_state=10)
param_test2 = {'max_depth':[5], 'min_samples_split':range(542,557,2)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1, min_samples_split=500,
                                                               min_samples_leaf=50,max_depth=8,max_features='sqrt',
                                                               subsample=0.8,random_state=10), 
                        param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=100)
gsearch2.fit(X_all_train,y_all_train)
print(gsearch2.cv_results_)
print(gsearch2.best_params_)
print(gsearch2.best_score_)
'''
###{'max_depth': 5, 'min_samples_split': 550} 0.861201787876
gbm3 = GradientBoostingClassifier(random_state=10)
param_test3 = {'max_features':range(1,10,1)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1, min_samples_split=550,
                                                               min_samples_leaf=50,max_depth=5,
                                                               subsample=0.8,random_state=10), 
                        param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=100)
gsearch3.fit(X_all_train,y_all_train)
print(gsearch3.best_params_)
print(gsearch3.best_score_)
###{'max_features': 4} 0.861201787876

param_test4 = {'min_samples_split':range(600,1300,100), 'min_samples_leaf':[90]}
gbm4 = GradientBoostingClassifier(random_state=10)
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1, 
                                                               max_depth=5, max_features=4,
                                                               subsample=0.8,random_state=10), 
                        param_grid = param_test4, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=100)
gsearch4.fit(X_all_train,y_all_train)
print(gsearch4.best_params_)
print(gsearch4.best_score_)
#{'min_samples_leaf': 90, 'min_samples_split': 550} 0.861394937547
param_test5 = {'learning_rate':np.arange(0.1,0.4,0.1)}
gbm5 = GradientBoostingClassifier(random_state=10)
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1, min_samples_leaf=90,
                                                               max_depth=5, max_features=4,min_samples_split=550,
                                                               subsample=0.8,random_state=10), 
                        param_grid = param_test5, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=100)
gsearch5.fit(X_all_train,y_all_train)
print(gsearch5.best_params_)
print(gsearch5.best_score_)
#{'subsample': 0.8}0.861394937547

y_hat = gsearch5.predict_proba(test)[:,1]
to_save = pd.DataFrame(id_test)
to_save['Y'] = y_hat
to_save.to_csv('gb_tuned2.csv',index=False)
