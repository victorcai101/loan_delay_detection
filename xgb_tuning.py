# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:49:18 2017

@author: victor
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from xgb_modelfit import modelfit
from sklearn import preprocessing
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.model_selection import StratifiedKFold
rcParams['figure.figsize'] = 12, 4

x_train = pd.read_csv('x_train.csv')
#x_test = pd.read_csv('stacking_test_output.csv')
#y_all_test = pd.read_csv('y_all_test.csv')
y_all_train = pd.read_csv('y_all_train.csv')
#test = data_test.loc[:,'F1':]
#test =  preprocessing.scale(test)
'''
xgb1 = XGBClassifier(learning_rate =0.001,
                     n_estimators=2000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'binary:logistic',
                     nthread=4,
                     scale_pos_weight=1)
cv_result = modelfit(xgb1, x_train, y_all_train)
'''
param_test1 = {
'reg_alpha':[0.001,0.005,0.01,0.05,0.1]
}
#from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=5)
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.01, n_estimators=240, max_depth=3,
                                                  min_child_weight=10, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', scale_pos_weight=1, seed=27,reg_alpha=0.01), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=kfold,verbose=100)

col_name = ['rf','gb','et']
gsearch1.fit(x_train.loc[:,col_name],y_all_train['Y'])
print(gsearch1.best_params_)
print(gsearch1.best_score_)
