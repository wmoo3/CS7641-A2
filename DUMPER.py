# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np

import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
#poker = pd.read_hdf('datasets.hdf','poker')       
#poker = pd.read_csv('./poker.csv')   
#pokerX = poker.drop('is_over_$1000',1).copy().values
#pokerY = poker['is_over_$1000'].copy().values

poker = pd.read_csv('./poker-hand.csv')  

'''
onehot=OneHotEncoder(categories='auto')
onehot.fit(poker.iloc[:,:10])
enc=pd.DataFrame(data=onehot.transform(poker.iloc[:,:10]).toarray(),columns=onehot.get_feature_names())

poker.drop(poker.iloc[:,:10],axis=1,inplace=True)
poker=poker.join(enc)
'''
 
pokerX = poker.drop('has_hand',1).copy().values
pokerY = poker['has_hand'].copy().values

poker_trgX, poker_tstX, poker_trgY, poker_tstY = ms.train_test_split(pokerX, pokerY, test_size=0.3, random_state=0,stratify=pokerY)     

#pipe = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),])
pipe = Pipeline([('Scale',StandardScaler()),])

trgX = pipe.fit_transform(poker_trgX,poker_trgY)
trgY = np.atleast_2d(poker_trgY).T
tstX = pipe.transform(poker_tstX)
tstY = np.atleast_2d(poker_tstY).T
trgX, valX, trgY, valY = ms.train_test_split(trgX, trgY, test_size=0.2, random_state=1,stratify=trgY)     
tst = pd.DataFrame(np.hstack((tstX,tstY)))
trg = pd.DataFrame(np.hstack((trgX,trgY)))
val = pd.DataFrame(np.hstack((valX,valY)))
tst.to_csv('ph_test.csv',index=False,header=False)
trg.to_csv('ph_trg.csv',index=False,header=False)
val.to_csv('ph_val.csv',index=False,header=False)