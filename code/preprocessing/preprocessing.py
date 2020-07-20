import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import warnings
import os
warnings.filterwarnings('ignore')

def DeriveFeature(data):
    # KD/A Feature
    data['blueKd/a'] = (data['blueKill'] + data['blueAssist']) / data['blueDeath']
    data['redKd/a'] = (data['redKill'] + data['redAssist']) / data['redDeath']

    # if Death Feature Value == 0 -> Generate NaN
    ## Processing NaN Value
    data.loc[~np.isfinite(data['blueKd/a']), 'blueKd/a'] = (data['blueKill'] + data['blueAssist']) / 1
    data.loc[~np.isfinite(data['redKd/a']), 'redKd/a'] = (data['redKill'] + data['redAssist']) / 1

    return data

def Preprocess(train_data, test_data, scaling, return_output = False, path = None):
    """ Preprocessing Dataset

    Parameters
    ----------
    train_data: {Pandas DataFrame}

    test_data: {Pandas DataFrame}

    scaling: {boolean} Scaling applicability

    return_output: {boolean}

    path: If return_output == True, input Save path.

    Returns
    -------
    X_train, X_test, y_train, y_test: {Numpy ndarray}
    """
    train = DeriveFeature(train_data)
    test = DeriveFeature(test_data)
  
    # One Hot Encoder
    dragon_type = train['blueDragnoType'].unique().tolist()
    true_value = dragon_type[:5]
    
    train = train.loc[(train['blueDragnoType'].isin(true_value)) & (train['redDragnoType'].isin(true_value))]
    train = train.loc[~((train['blueDragnoType'] != "[]") &(train['redDragnoType'] != '[]'))].reset_index(drop = True)

    test = test.loc[(test['blueDragnoType'].isin(true_value)) & (test['redDragnoType'].isin(true_value))]
    test = test.loc[~((test['blueDragnoType'] != "[]") &(test['redDragnoType'] != '[]'))].reset_index(drop = True)
    # Create Interactive Categorical Feature
    ## About Train Dataset
    ### Dragon Kill
    train['dragonKill'] = train['blueDragnoType'] + train['redDragnoType']
    train.drop(columns = ['blueDragnoType', 'redDragnoType', 'blueDragon', 'blueFirstDragon', 'redDragon', 'redFirstDragon'], inplace = True)

    dragon_encoder = OneHotEncoder()
    dragon_cat = dragon_encoder.fit_transform(train['dragonKill'].values.reshape(-1, 1))
    dragon_cat_cols = dragon_encoder.get_feature_names('D')
    dragon_df = pd.DataFrame(dragon_cat.todense(), columns = dragon_cat_cols)

    train = pd.concat([train.drop(columns = 'dragonKill'), dragon_df], axis = 1)

    ### Tower Kill
    train['FirstKillLane'] = train['blueFirstTowerLane'] + train['redFirstTowerLane']
    train.drop(columns = ['blueFirstTowerLane', 'redFirstTowerLane'], inplace = True)
    
    tower_encoder = OneHotEncoder()
    tower_cat = tower_encoder.fit_transform(train['FirstKillLane'].values.reshape(-1, 1))
    tower_cat_cols = tower_encoder.get_feature_names('T')
    tower_df = pd.DataFrame(tower_cat.todense(), columns = tower_cat_cols)

    train = pd.concat([train.drop(columns = 'FirstKillLane'), tower_df], axis = 1)
    
    ## About Test Dataset
    ### Dragon Kill
    test['dragonKill'] = test['blueDragnoType'] + test['redDragnoType']
    test.drop(columns = ['blueDragnoType', 'redDragnoType', 'blueDragon', 'blueFirstDragon', 'redDragon', 'redFirstDragon'], inplace = True)
    
    dragon_cat = dragon_encoder.fit_transform(test['dragonKill'].values.reshape(-1, 1))
    dragon_cat_cols = dragon_encoder.get_feature_names('D')
    dragon_df = pd.DataFrame(dragon_cat.todense(), columns = dragon_cat_cols)

    test = pd.concat([test.drop(columns = 'dragonKill'), dragon_df], axis = 1)
    
    ### Tower Kill
    test['FirstKillLane'] = test['blueFirstTowerLane'] + test['redFirstTowerLane']
    test.drop(columns = ['blueFirstTowerLane', 'redFirstTowerLane'], inplace = True)
    
    tower_cat = tower_encoder.transform(test['FirstKillLane'].values.reshape(-1, 1))
    tower_cat_cols = tower_encoder.get_feature_names('T')
    tower_df = pd.DataFrame(tower_cat.todense(), columns = tower_cat_cols)

    test = pd.concat([test.drop(columns = 'FirstKillLane'), tower_df], axis = 1)
    
    # PostProcessing
    regex = re.compile(r"\[]", re.IGNORECASE)
    train.columns = [regex.sub("_None_", col) if any(x in str(col) for x in set(('[]'))) else col for col in train.columns.values]
    test.columns = [regex.sub("_None_", col) if any(x in str(col) for x in set(('[]'))) else col for col in test.columns.values]

    regex = re.compile(r"\[|\]", re.IGNORECASE)
    train.columns = [regex.sub("", col) if any(x in str(col) for x in set(('[]'))) else col for col in train.columns.values]
    test.columns = [regex.sub("", col) if any(x in str(col) for x in set(('[]'))) else col for col in test.columns.values]
    
    # Scaling
    train = train.set_index('gameId')
    test = test.set_index('gameId')

    X_train = train.drop(columns = 'blueWins')
    y_train = train['blueWins']

    X_test = test.drop(columns = 'blueWins')
    y_test = test['blueWins']

    print(X_train.shape)
    print(X_test.shape)
    if scaling == True:
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif scaling == False:
        pass
    
    if return_output == True:
        pass

    else:
        pass

    return X_train, X_test, y_train, y_test