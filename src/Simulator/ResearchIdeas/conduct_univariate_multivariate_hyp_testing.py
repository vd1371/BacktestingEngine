import os
import numpy as np
import pandas as pd
import pprint

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump

from config import LONG, SHORT

from .reseach_utils import clean_nans
from .reseach_utils import convert_categorical_to_binary

import logging
logger = logging.getLogger("research_logger")

P_VALUE_THRESHOLD = 0.1
TEST_SIZE = 0.4

def conduct_univariate_multivariate_hyp_testing(df, **params):

    '''
    ## GUIDE: Step 12

    This is an example of how we can use the univariate
    Hypothesis testing to improve a strategy.
    But the below code is written in the class and can be
    significantly better. This is just a starting point for students.

    Args:
        df: pd.DataFrame
            The dataframe of the potential trades

    Returns:
        None
    '''

    # df = df.copy()
    # df.dropna(inplace=True, axis=0)
    # Split into train and test
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=42)

    short_trades = df_train[df_train['trade_direction'] == SHORT]
    long_trades = df_train[df_train['trade_direction'] == LONG]


    # Univariate Hypothesis Testing on the long ones
    holder = {}
    for col in long_trades.columns:

        if not col.startswith("stat_"):
            continue

        df_tmp = long_trades[[col, 'PnL_ratio']].copy()
        df_tmp.dropna(inplace=True, axis=0)

        X = df_tmp[col]
        Y = df_tmp['PnL_ratio']
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()
        p_value = model.pvalues[col]
        holder[col] = p_value

    print ("\n\np-values of stats for Long Trades")
    print ("---------------------------------")
    df = pd.DataFrame.from_dict(holder, orient='index', columns=['p_value'])
    print (df)

    # Univariate Hypothesis Testing on the short ones
    holder = {}
    for col in short_trades.columns:

        if not col.startswith("stat_"):
            continue

        df_tmp = short_trades[[col, 'PnL_ratio']].copy()
        df_tmp.dropna(inplace=True, axis=0)

        X = df_tmp[col]
        Y = df_tmp['PnL_ratio']
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()
        p_value = model.pvalues[col]
        holder[col] = p_value

    print ("\n\np-values of stats for Short Trades")
    print ("---------------------------------")
    df = pd.DataFrame.from_dict(holder, orient='index', columns=['p_value'])
    print (df)

    # Regress PnL vs stat_Vola(22)_1d
    df_tmp = short_trades[['stat_Vola(22)_1d', 'PnL_ratio']].copy()
    df_tmp.dropna(inplace=True, axis=0)
    X = df_tmp[col]
    Y = df_tmp['PnL_ratio']
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()
    print (model.summary())

    return