import os
import pandas as pd
import time

from ._add_statistical_measures import add_statistical_measures
from ._add_technical_indicators import add_technical_indicators
from src.Simulator.Signals import get_alpha_signal_func


def combine_data_add_signal(
        macro_and_other_data,
        **params
    ):
    data_local = macro_and_other_data['data']


    df, is_updated = add_technical_indicators(data_local, **params)
    df, is_updated = add_statistical_measures(
        df,
        macro_and_other_data=macro_and_other_data,
        interval = "1d",
        **params
        )
    
    for col in df.columns:
        if col in ['Open', 'High', 'Low', 'Close']:
            continue
        df[col] = df[col].ffill()

    alpha_signal_func = get_alpha_signal_func(**params)
    df = alpha_signal_func(df, **params)

    # The following lines are used in the run_alpha_strategy
    df['Close_t_1'] = df['Close'].shift()

    if 'close_short_signal' not in df.columns:
        df['close_short_signal'] = 0

    if 'close_long_signal' not in df.columns:
        df['close_long_signal'] = 0

    # Add the end of candle flag
    df = add_end_of_candle_flag(df, **params)

    return df.copy()


def add_end_of_candle_flag(df, **params):
    '''
    This function is used to find the end of candle
    '''
    df['end_of_candle'] = 0

    should_close_at_end_of_candle = params['should_close_at_end_of_candle']
    if not should_close_at_end_of_candle:
        return df
    
    df['end_of_candle'] = 1

    return df