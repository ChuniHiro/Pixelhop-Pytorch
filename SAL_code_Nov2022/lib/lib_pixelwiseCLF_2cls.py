#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pixel-wise classification
@Author: Yijing Yang
@Date: 2021.07.08
@Contact: yangyijing710@outlook.com
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from lib.lib_tuning import tune_LR, tune_xgb
from sklearn.preprocessing import StandardScaler

class pixelwiseCLF():
    def __init__(self, numcls=2, regC=0.01, standardize=True, model_select=False, GPU = None):
        self.numcls = numcls
        self.regC = regC
        self.GPU = GPU
        self.standardize = standardize
        # if self.standardize == True:
        #     self.scaler = StandardScaler()
        self.model_select = model_select
        
        
    def fit(self, X, y, sample_weight=None, chosen=None):
        self.trN, self.H, self.W, self.C = X.shape

        # flatten for pixel-wise classification
        X = self.flatten(X)
        y_pix = np.repeat(y, self.H*self.W)

        if chosen is None:
            if sample_weight is None:
                self.SI = None
            else:
                self.SI = sample_weight.reshape(-1)
        else:
            chosen = chosen.reshape(-1)
            X = np.copy(X[chosen])
            y_pix = np.copy(y_pix[chosen])
            if sample_weight is None:
                self.SI = None
            else:
                self.SI = sample_weight.reshape(-1)[chosen]

            print('selected train num = {}'.format(np.sum(chosen)/chosen.size))
            print(y_pix.size)
            print(chosen.size)


        if self.standardize == True:
            self.scaler = StandardScaler(with_mean=False).fit(X)
            X_ = self.scaler.transform(X)
        else:
            X_ = np.copy(X)
        
        if self.model_select == True:
            self.clf = tune_LR(X_, y_pix, params=None, folds=5, param_comb=10)
            # self.clf = tune_xgb(X_, y_pix, GPU=self.GPU, params=None, folds=5, param_comb=10)
        else:
            # self.clf = LogisticRegression(n_jobs=8, C=self.regC, solver='saga', class_weight='balanced').fit(X_, y_pix.reshape(-1))
            if self.GPU is None:
                print('gpu is None')
                self.clf = xgb.XGBClassifier(n_jobs=8,
                                             objective='binary:logistic',
                                             # tree_method='gpu_hist', gpu_id=self.GPU,
                                             max_depth=6, n_estimators=300,
                                             min_child_weight=5, gamma=5,
                                             subsample=0.8, learning_rate=0.1,
                                             nthread=8, colsample_bytree=1.0).fit(X_, y_pix, sample_weight=self.SI)
            else:
                self.clf = xgb.XGBClassifier(n_jobs=8,
                                             objective='binary:logistic',
                                             tree_method='gpu_hist', gpu_id=self.GPU,
                                            max_depth=6, n_estimators=300,
                                            min_child_weight=5, gamma=5,
                                            subsample=0.8, learning_rate=0.1,
                                            nthread=8, colsample_bytree=1.0).fit(X_, y_pix, sample_weight=self.SI)

        print('finish fitting...')
        
        
    def predict_proba(self, feat_X, y=None, reduce=0):
        self.N, _,_,_ = feat_X.shape
        feat_X = self.flatten(feat_X)
        
        if self.standardize == True:
            feat_X_ = self.scaler.transform(feat_X)
        else:
            feat_X_ = np.copy(feat_X)
            
        proba = self.clf.predict_proba(feat_X_)
       
        if reduce==1:
            proba_ = proba[:,1:]
        else:
            proba_ = np.copy(proba)
        
        if y is None:
            return self.expand(proba_), 0.0
        else:
            print('evaluate')
            y_pix = np.repeat(y, self.H*self.W)
            acc = 1-self.evaluate(proba, y_pix, 'error')
            return self.expand(proba_), acc
        
        
    def flatten(self, X):
        return X.reshape(-1, self.C)
        
    def expand(self, X):
        return X.reshape(self.N, self.H, self.W, -1)
        
    def evaluate(self, y_prob, gt, eval_metric):
        loss=0
        if eval_metric =='logloss':
            loss = log_loss(gt, y_prob)
        elif eval_metric=='error':
            loss = np.sum(np.argmax(y_prob,axis=1)!=gt)/gt.size
        return loss    
        
    
    