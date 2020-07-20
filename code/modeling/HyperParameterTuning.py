import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

import time
import gc
import os
import warnings
warnings.filterwarnings('ignore')

def RandomForestEvaluation(params):
    params = {'n_estimators' : int(params['n_estimators']),
        'max_depth' : int(params['max_depth']),
        'min_samples_leaf' : int(params['min_samples_split']),
        'max_leaf_nodes' : int(params['max_leaf_nodes'])}

    start_time = time.time()
    print('#' * 20, 'New Run', '#' * 20)
    print(f"params = {params}")
    
    k = 5
    step = 1
    
    kf = KFold(n_splits = k, random_state = 42)
    score_mean = 0

    for tr_idx, val_idx in kf.split(X_train, y_train):
        clf = RandomForestClassifier(**params, random_state = 42)

        X_tr, X_vl = X_train[tr_idx], X_train[val_idx]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_vl)

        score = accuracy_score(y_vl, pred)
        score_mean += score
        print(f'{step} CV - Score: {round(score, 4)}')

        step += 1
    
    end_time = time.time() - start_time
    print(f'Total Run Time :{round(end_time/60,2)}')

    gc.collect()
    print("Bayesian Optimization RandomForest Model Mean Accuracy".format(score_mean/k))

    del X_tr, X_vl, y_tr, y_vl, clf, score

    return (score_mean / k)


def XGBEvaluation(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma' : "{:.3f}".format(params['gamma']),
        'subsample' : "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'learning_rate' : "{:.3f}".format(params['learning_rate']),
        'num_leaves' : "{:.3f}".format(params['num_leaves']),
        'colsample_bytree' : "{:.3f}".format(params['colsample_bytree']),
        'min_child_samples' : "{:.3f}".format(params['min_child_samples']),
        'feature_fraction' : "{:.3f}".format(params['feature_fraction']),
        'bagging_fraction' : "{:.3f}".format(params['bagging_fraction'])
    }

    start_time = time.time()
    print('#' * 20, 'New Run', '#' * 20)
    print(f"params = {params}")
    
    k = 5
    step = 1

    kf = KFold(n_splits = k, random_state = 42)
    score_mean = 0
    
    for tr_idx, val_idx in kf.split(X_train, y_train):
        clf = XGBClassifier(**params, random_state = 42)

        X_tr, X_vl = X_train[tr_idx], X_train[val_idx]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        clf.fit(X_tr, y_tr, eval_set = [(X_tr, y_tr), (X_vl, y_vl)], early_stopping_rounds = 50, verbose = False)
        pred = clf.predict(X_vl)

        score = accuracy_score(y_vl, pred)
        score_mean += score
        print(f'{step} CV - Score: {round(score, 4)}')

        step += 1
    
    end_time = time.time() - start_time
    print(f'Total Run Time :{round(end_time/60,2)}')

    gc.collect()
    print("Bayesian Optimization XGBoost Model Mean Accuracy".format(score_mean/k))

    del X_tr, X_vl, y_tr, y_vl, clf, score

    return (score_mean / k)
