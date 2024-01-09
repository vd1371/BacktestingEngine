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

    # TODO: implement this function
    return