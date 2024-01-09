import os
import pandas as pd

import datetime

from pprint import pprint
from src.utils import convert_index_to_datetime
from config import get_today

def load_data_from_cache(symbol, cache_dir, **params):

    trading_interval = params['trading_interval']

    years = params['years_to_consider']
    market = params['market']

    file_path = os.path.join(
        cache_dir,
        f"{symbol}-{trading_interval}-{years}.csv"
    )

    loaded_from_cache = False
    df = None
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col= 0)
        df.index = convert_index_to_datetime(df, market, trading_interval)

        loaded_from_cache = True

    return loaded_from_cache, df

def cache_exists(symbol, cache_dir, **params):

    trading_interval = params['trading_interval']

    years = params['years_to_consider']
    market = params['market']

    file_path = os.path.join(
        cache_dir,
        f"{symbol}-{trading_interval}-{years}.csv"
    )

    return os.path.exists(file_path)


def save_data_to_cache(df, symbol, cache_dir, **params):

    trading_interval = params['trading_interval']
    years = params['years_to_consider']
    market = params['market']

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_path = os.path.join(
        cache_dir,
        f"{symbol}-{trading_interval}-{years}.csv"
    )

    df.to_csv(file_path)

    return