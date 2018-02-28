# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:00:35 2017

@author: victor
"""

# Load in our libraries
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
#test =  preprocessing.scale(test)

ntrain = X_all_train.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

'''
# Cross validate model with Kfold stratified cross val
#kfold = StratifiedKFold(n_splits=10)
kfold = 5
# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(SVC(random_state=random_state))

cv_results = []
for classifier in classifiers :
    print(classifier)
    cv_results.append(cross_val_score(classifier, X_all_train, y_all_train, scoring = "roc_auc", cv = kfold))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["RandomForest","DecisionTree","KNeighboors","AdaBoost",
"ExtraTrees","GradientBoosting","MultipleLayerPerceptron","LogisticRegression","LinearDiscriminantAnalysis","SVC"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
'''
#Random Forest Parameters
rf_params = {
        #'n_jobs': -1,
        'n_estimators': 180,
        #'warm_start': True, 
        'max_features': 'auto',
        'max_depth': 14,
        'min_samples_leaf': 25,
        #'max_features' : 'sqrt',
        #'verbose': 0
}

# Extra Trees Parameters
et_params = {"max_depth": None,
              "max_features": 10,
              "min_samples_split": 15,
              "min_samples_leaf": 10,
              "bootstrap": False,
              "n_estimators" :200,
              "criterion": "gini"}
# AdaBoost parameters
ada_params = {
        'n_estimators': 200,
        'learning_rate' : 0.1,
}

# Gradient Boosting parameters
gb_params = {
        'n_estimators': 50,
        'learning_rate':0.1,
        'max_features': 4,
        'max_depth': 5,
        'min_samples_leaf': 90,
        'min_samples_split':550,
        'subsample':0.8,
}

# Support Vector Classifier parameters 
SEED = 1
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
#svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
X_all_train = X
y_all_train = y
X_all_test = test
print('training adaboost')
ada_oof_train, ada_oof_test, ada = get_oof(ada, X_all_train, y_all_train, X_all_test,NFOLDS,kf) # AdaBoost 
#print('adaboost training score:',roc_auc_score(y_all_test,ada_oof_test))

print('training extra trees')
et_oof_train, et_oof_test, et = get_oof(et, X_all_train, y_all_train, X_all_test,NFOLDS,kf) # Extra Trees
#print('extra tree training score:',roc_auc_score(y_all_test,et_oof_test))

print('training random forest')
rf_oof_train, rf_oof_test, rf = get_oof(rf,X_all_train, y_all_train,X_all_test,NFOLDS,kf) # Random Forest
#print('random forest training score:',roc_auc_score(y_all_test,rf_oof_test))
print('training gradient boost')
gb_oof_train, gb_oof_test, gb = get_oof(gb,X_all_train, y_all_train, X_all_test,NFOLDS,kf) # Gradient Boost
#print('gradient boost training score:',roc_auc_score(y_all_test,gb_oof_test))

print("Training is complete")

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
n_fold = StratifiedKFold(n_splits=3)
parameters = {"min_weight_fraction_leaf":[0.0005]}
#"min_samples_leaf":25
#"min_samples_split":[100]
RFC = RandomForestClassifier(random_state=50,max_depth=5,n_estimators=200, 
                             max_features = "auto", min_samples_leaf=25, 
                             min_samples_split=500,min_weight_fraction_leaf=0.0005)
#clf = GridSearchCV(RFC,parameters,cv=5,scoring=auc_scorer)
clf = GridSearchCV(RFC,parameters,cv=n_fold,scoring='roc_auc', n_jobs = 1,verbose=100)
clf.fit(x_train,y_all_train)
y_hat = clf.predict_proba(x_test)[:,1]

to_save = pd.DataFrame(id_test)
to_save['Y'] = y_hat
to_save.to_csv('aa_stacking_rf.csv',index=False)
#print(clf.best_params_)
#print(clf.best_score_)
