import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import warnings
import os
warnings.filterwarnings('ignore')

def Load_Data(path, minute, return_test, split_size = 0.25):
    """ Load data from parameter 'path'

    Parameters
    ----------
    path: {str} Path where data exists

    minute: {int (10 or 15)} Get 10 minutes and 15 minutes of data after game starts.

    return_test: {boolean} Whehers to split the data into training data and test data.

    test_size: {float} if return_test == True, data split ratio.


    Returns
    -------
    If retrun_test == True, return train, test. 

       return_test == False, return data.
    """
    if minute == 10:
        data_dir = os.path.join(path, 'Challenger_Ranked_Games_10minute.csv')

    elif minute == 15:
        data_dir = os.path.join(path, 'Challenger_Ranked_Games_15minute.csv')

    data = pd.read_csv(data_dir)
    data.drop(columns = 'redWins', inplace = True)

    if return_test == True:
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = 'blueWins'), data['blueWins'],
        test_size = split_size, random_state = 42)

        train = pd.concat([X_train, y_train], axis = 1).reset_index(drop = True)
        test = pd.concat([X_test, y_test], axis = 1).reset_index(drop = True)

        print("Train Dataset Shape: ", train.shape)
        print("Test Dataset Shape: ", test.shape)
        return train, test
    
    else:
        print("Dataset Shape: ", data.shape)
        return data