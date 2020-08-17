import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE, RFECV

import warnings

class FeatureSelection:
    def __init__(self, model, train_features, train_label, test_features, test_label):
        self.model = model
        self.X_train = train_features
        self.y_train = train_label
        self.X_test = test_features
        self.y_test = test_label
        
    def RecursiveFeatureElimination(self, n_features, step):
        accuracy_list = []
        
        if type(n_features) == list:
            for i in n_features:
                print("Feature Selection Algorithm Start... \nNumber of Features: {} | Elimination Step: {}".format(i, step))
                rfe_selector = RFE(self.model, i, step = step, verbose = 1)
                rfe_model = rfe_selector.fit(self.X_train, self.y_train)
                pred = rfe_selector.predict(self.X_test)
                
                acc_score = accuracy_score(self.y_test, pred)
                accuracy_list.append(acc_score)
                print("Accuracy Score: ", acc_score)
            
            accuracy_array = np.array(accuracy_list)
            max_accuracy_index = int(np.where(accuracy_array == max(accuracy_array))[0])
            print("\nMax Accuracy Score: {} | Number of Features: {}".format(max(accuracy_array), n_features[max_accuracy_index]))
            
    def RecursiveFeatureEliminationCV(self, step, cv):
        self.step = step
        self.cv = cv
        
        rfecv_selector = RFECV(self.model, step = self.step, cv = self.cv)
        rfecv_model = rfecv_selector.fit(self.X_train, self.y_train)
        
        print("Num Features: %d" % rfecv_model.n_features_)
        print("Selected Features: %s" % rfecv_model.support_)
        print("Feature Ranking: %s" % rfecv_model.ranking_)