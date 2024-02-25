import os
import pandas as pd
import datetime
import logging

import yfinance as yf

from ._raw_data_cache_handler import load_data_from_cache
from ._raw_data_cache_handler import save_data_to_cache

from src.utils import convert_index_to_datetime

def load_data_from_yahoo_finance(
    symbol, 
    market_macro_index = False,
    **params):

    '''
    Load the data from Yahoo Finance
    If the data is not in the cache, it loads the data from Yahoo Finance and saves it to the cache.
    If the data is in the cache, it loads the data from the cache.

    Args:
    symbol: str
        The symbol of the stock

    market_macro_index: bool
        If the data is for the market macro index
    
    params: dict
        The parameters for the data

    Returns:
        df: pd.DataFrame
    
    '''

    market = params['market']
    trading_interval = params.get("trading_interval")
    years = params.get("years_to_consider")
    last_date_for_strategy = params.get("last_date_for_strategy", datetime.datetime.today())

    today = last_date_for_strategy.strftime("%Y%m%d")
    logging.info("Last date of simulation was set as %s"%today)

    loaded_from_cache, df = load_data_from_cache(symbol, **params)
    
    if not loaded_from_cache:
        if market_macro_index: # When we are getting the index prices
            # We only need the daily values of the market
            trading_interval = "1d"
            df = _get_data_from_yahoo_finance(symbol, market, trading_interval, years+2, last_date_for_strategy)
        else:
            df = _get_data_from_yahoo_finance(symbol, market, trading_interval, years, last_date_for_strategy)

        # Save the data to cache
        save_data_to_cache(df, symbol, **params)
    
    return df


def _get_data_from_yahoo_finance(symbol, market, trading_interval, years, last_date_for_strategy):

    '''
    Get the data from Yahoo Finance
    it loads the data from Yahoo Finance
    
    Args:
    symbol: str
        The symbol of the stock

    market: str
        The market of the stock
    
    trading_interval: str
        The trading interval of the stock

    years: int
        The number of years to consider

    last_date_for_strategy: datetime.datetime
        The last date for the strategy
    '''

    stock_ticker = yf.Ticker(symbol)
    df = stock_ticker.history(
        # period = "%dy"%years,
        # "- 1" below prevents YF from complaining about data unavailability for HK hourly data;
        # without the - 1, it would raise:
        #       1299.HK: 1h data not available for startTime=1628783383 and endTime=1691884184.
        #       The requested range must be within the last 730 days.
        start = last_date_for_strategy - datetime.timedelta(days=int(years*365) - 1),
        end = last_date_for_strategy + datetime.timedelta(days=1),  # We need to add 1 day to get the last day
        interval = trading_interval,
    )

    # Correct the timezone
    df.index = convert_index_to_datetime(df, market, trading_interval)

    return df