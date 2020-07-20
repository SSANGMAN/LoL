import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import os
import warnings
warnings.filterwarnings('ignore')

def KFoldValidation(X, y, algorithm, k):
    """ KFold Cross Validation

    Parameters
    ==========
    X: {Numpy ndarray}
    
    y: {Numpy ndarray}

    algorithm: {str} Select Classification Algorithm
        implemented Algorithm
        1. LogisticRegression
        2. DecisionTree
        3. XGB
        4. RandomForest

    """
    kf = KFold(n_splits = k, random_state = 42)
    accuracy = []
    step = 1

    for tr_idx, val_idx in kf.split(X,y):
        print('*' * 30, 'New Fold', '*' * 30)
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        if algorithm == 'LogisticRegression':
            clf = LogisticRegression(random_state = 42) 
        elif algorithm == 'DecisionTree':
            # max_depth = 7: Manual Search
            clf = DecisionTreeClassifier(max_depth = 7,random_state = 42)
        elif algorithm == 'XGB':
            clf = XGBClassifier(max_depth = 6, random_state = 42)
        elif algorithm == 'RandomForest':
            clf = RandomForestClassifier(max_depth = 7, random_state = 42)
            
        if algorithm == 'XGB':
            clf.fit(X_tr, y_tr, eval_set = [(X_tr, y_tr), (X_val, y_val)],
                   early_stopping_rounds = 50, verbose = 10)
        else:
            clf.fit(X_tr, y_tr)
            
        pred = clf.predict(X_val)
        print("\n{}-Fold {} Accuracy Score: {}".format(step, algorithm, accuracy_score(pred, y_val)))
        
        accuracy.append(accuracy_score(pred, y_val))
        del X_tr, y_tr, X_val, y_val, clf
    
        step += 1
        if step == k:
            print("*" * 20, "End Validation", "*" * 20)
            
    print("\nMean Accuracy Score:, ", sum(accuracy) / k)