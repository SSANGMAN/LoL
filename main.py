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

import joblib
import time
import gc
import os
import warnings
import sys
sys.path.append("./code")
warnings.filterwarnings('ignore')

from dataset.load import Load_Data
from preprocessing.preprocessing import DeriveFeature, Preprocess
from modeling.Validation import KFoldValidation
from modeling.HyperParameterTuning import RandomForestEvaluation, XGBEvaluation

path = "./dataset"
model_output_path = "./model"
als = ['LogisticRegression', 'DecisionTree', 'XGB', 'RandomForest']

algorithm = 'XGB'
k = 5

# 1. Load Data
train, test = Load_Data(path, minute = 10, return_test = True, split_size = 0.25)

# 2. Preprocessing
X_train, X_test, y_train, y_test = Preprocess(train, test, scaling = True)

# 3. Basic Modeling
#for al in als:
#    KFoldValidation(X_train, y_train, algorithm = al, k = 5)


# 4. HyperParameters Tuning
if algorithm == 'RandomForest':
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

        return -(score_mean / k)

    space = {'n_estimators' : hp.quniform('n_estimators', 100, 500, 25),
                'max_depth': hp.choice('max_depth', list(range(10,20,2))),
                'min_samples_leaf': hp.choice('min_samples_leaf', list(range(10,100,100))),
                'min_samples_split' : hp.choice('min_samples_split', list(range(10,50,10))),
                'max_leaf_nodes' : hp.choice('max_leaf_nodes', list(range(20,150,10)))
    }

elif algorithm == 'XGB':
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
            clf = XGBClassifier(**params, tree_method = 'gpu_hist', predictor = 'gpu_predictor', random_state = 42, objective='binary:logistic', eval_metric = 'error')

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
        print("Bayesian Optimization XGBoost Model Mean Accuracy: {}".format(score_mean/k))

        del X_tr, X_vl, y_tr, y_vl, clf, score

        return -(score_mean / k)

    space = {
    'max_depth' : hp.quniform('max_depth', 3, 10, 1),
    'reg_alpha' : hp.uniform('reg_alpha', 0.01, 0.4),
    'reg_lambda': hp.uniform('reg_lambda', 0.6, 1),
    'learning_rate' : hp.uniform('learning_rate', 0.01, 0.2),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.3, 0.9),
    'gamma' : hp.uniform('gamma', 0.01, .7),
    'num_leaves' : hp.choice('num_leaves', list(range(10,100,10))),
    'min_child_samples': hp.choice('min_child_samples', list(range(0,100,10))),
    'subsample': hp.choice('subsample', [.5, .6, .7, .8, .9, 1]),
    'feature_fraction' : hp.uniform('feature_fraction', .5, .8),
    'bagging_fraction' : hp.uniform('bagging_fraction', .5, .9)
}

if algorithm == 'RandomForest':
    best = fmin(fn = RandomForestEvalation, space = space, algo = tpe.suggest, max_eval = 50)
elif algorithm == 'XGB':
    best = fmin(fn = XGBEvaluation, space = space, algo = tpe.suggest, max_evals = 50)

best_params = space_eval(space, best)
print("Best HyperParameters: ", best_params)
best_params['max_depth'] = int(best_params['max_depth'])

## Training Model
if algorithm == 'RandomForest':
    clf = RandomForestClassifier(**best_params, random_state = 42)
elif algorithm == 'XGB':
    clf = XGBClassifier(**best_params, tree_method = 'gpu_hist', predictor = 'gpu_predictor', random_state = 42, objective='binary:logistic', eval_metric = 'error')

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("HyperParameter Tuning XGBClassifier Accuracy Score: ", accuracy_score(pred, y_test))

# Save Model
joblib.dump(clf, os.path.join(model_output_path, 'XGB_{}'.format(round(accuracy_score(pred, y_test), 3))))