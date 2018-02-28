# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 23:04:02 2017

@author: victor
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
data_train = pd.read_csv('train_cleaned_v1.csv')
data_test = pd.read_csv('test_cleaned_v1.csv')

id_test = pd.DataFrame(data_test['id'])
y = data_train.loc[:,'Y']
X = data_train.loc[:,'F1':]

all_features = pd.concat([data_train.loc[:,'F1':],data_test.loc[:,'F1':]])

cate_name = ['F1','F2','F4','F5','F7','F8','F10','F12','F14','F15','F17','F20','F24','F25']
cont_name = ['F6','F9','F11','F16','F18','F19','F21','F22','F23','F27']
#cate_name = ['F1','F2','F4','F5','F7','F8','F10','F12','F14','F15','F17','F20','F24','F25']
#nogroup_name = ['F6','F9','F16','F19','F21','F22','F27']
#nogroup_dum_name = ['F1','F4','F5','F7','F8','F10','F12','F13','F15','F17','F20','F24']
#group1_name = ['F3','F23']
#group2_name = ['F18','F26']
#group_dum_name = ['F2','F14','F25']
#group_dum = cate.loc[:,group_dum_name]

cate = all_features.loc[:,cate_name]
cont = all_features.loc[:,cont_name]
cont = preprocessing.scale(cont)
cont = pd.DataFrame(cont,columns=cont_name)

cate = pd.get_dummies(cate)
'''
group1 = np.array(cont.loc[:,group1_name])
group2 = np.array(cont.loc[:,group2_name])
no_group = cont.loc[:,nogroup_name]
#group_dum = cate.loc[:,group_dum_name]
#nogroup_dum = cate.loc[:,nogroup_dum_name]
pca1 = PCA(n_components = 1)
F3_23 = pca1.fit_transform(group1)
F3_23 = pd.DataFrame(F3_23,columns=['pca_3_23'])
pca2 = PCA(n_components = 1)
F18_26 = pca2.fit_transform(group2)
F18_26 = pd.DataFrame(F18_26,columns=['pca_18_26'])

pca3 = PCA(n_components = 1)
#F_dum = pd.DataFrame(pca3.fit_transform(group_dum),columns=['pca_2_14_15'])
#F_dum = pd.get_dummies(F_dum)


#nogroup_dum = pd.get_dummies(nogroup_dum)
no_group.index = F_dum.index
#nogroup_dum.index = F_dum.index
'''
cont.index = cate.index
proc_features = pd.concat([cont,cate],axis=1)



proc_train = proc_features[0:X.shape[0]]
proc_train.index = y.index
proc_train = pd.concat([y,proc_train],axis=1)
proc_test = proc_features[49998:]
proc_test.index=id_test.index
proc_test = pd.concat([id_test,proc_test],axis=1)


proc_train.to_csv('train_cleaned_v12_with11.csv',index=False)
proc_test.to_csv('test_cleaned_v12_with11.csv',index=False)

'''
dum = pd.get_dummies(cate['F1'],prefix='F1')
remain_cate = ['F2','F4','F5','F7','F8','F10','F12','F13','F14','F15','F17','F20','F24','F25']
for i in remain_cate:
    dum = pd.concat([dum,pd.get_dummies(cate[i],prefix=i)],axis=1,join_axes=[dum.index])
'''
'''
j=0
freq_dum_name = []
for i in dum.columns:
    if dum[i].sum()>100:
        freq_dum_name.append(i)
        j=j+1
print(j)
'''