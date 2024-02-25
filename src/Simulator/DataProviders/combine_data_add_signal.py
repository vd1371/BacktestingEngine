import os
import pandas as pd
import time

from ._add_statistical_measures import add_statistical_measures
from ._add_technical_indicators import add_technical_indicators
from src.Simulator.Signals import get_alpha_signal_func
from src.Simulator.Signals import get_universe_signal_func


def combine_data_add_signal(
        macro_and_other_data,
        **params
    ):
    '''
    Combine the data and add the signal

    This function combines the data and adds the signal to the data.
    It uses the add_technical_indicators and add_statistical_measures functions
    to add the technical indicators and statistical measures to the data.
    Then, it uses the get_alpha_signal_func to get the signal function and
    adds the signal to the data.

    get_alpha_signal_func is a function that returns the signal function based on the parameters.
    The signal function is used to add the signal to the data.

    Args:
        macro_and_other_data: dict
            The macro and other data

        **params: dict

    Returns:
        df: pd.DataFrame
    
    '''

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

    if get_universe_signal_func(**params) is None:
        alpha_signal_func = get_alpha_signal_func(**params)
        df = alpha_signal_func(df, **params)

    # The following lines are used in the run_alpha_strategy
    df['Close_t_1'] = df['Close'].shift()

    # Add the close short and close long signals to the data
    # These signals are used to close the trade at the end of the candle when
    # the should_close_at_end_of_candle is True and close_long_signal or
    # close_short_signal is 1. You may add these signals to the data in your
    # signal function. This is for the cases when the user has not added these
    # signals to the data in the signal function.

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

    If the should_close_at_end_of_candle is True, then it adds the end_of_candle flag to the data.
    The end_of_candle flag is used to close the trade at the end of the candle.

    Args:
        df: pd.DataFrame
            The data

        **params: dict

    Returns:
        df: pd.DataFrame
    '''
    df['end_of_candle'] = 0

    should_close_at_end_of_candle = params['should_close_at_end_of_candle']
    if not should_close_at_end_of_candle:
        return df
    
    df['end_of_candle'] = 1

    return df