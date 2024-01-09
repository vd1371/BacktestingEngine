import pandas as pd
import numpy as np

from ._add_arima_forecasting import _add_arima_forecasting
from ._add_garch_forecasting import _add_garch_forecasting

def add_statistical_measures(
        df,
        macro_and_other_data = None,
        interval = None,
        **params):
    
    '''
    macro_and_other_data: dict

        It consists of information provided from the AllStockPrices class.
        It contains the dataframes of the other stocks, the market index.
        It is usually passed for the daily data, so that we can use the daily data of the other indices
    '''

    ## TODO: ASSIGNMENT #3: Add Beta and IV here

    # This flag will be used to check if we need to save the data to cache
    is_updated = False
    suffix = "" if interval is None else f"_{interval}"
    initial_length_of_columns = len(df.columns)

    for t in [22]:
        if f"stat_Vola({t}){suffix}" not in df:
            df[f"stat_Vola({t}){suffix}"] = df['Close'].shift().pct_change().rolling(window=t).std()*(252**0.5)

    if macro_and_other_data is not None:
        # For adding the data related to the macro and market index
        
        for k, data in macro_and_other_data.items():
            if k in ['data', 'symbol', 'cache_dir']:
                continue
            
            if f'stat_{k}_change(t_1)_ratio{suffix}' not in df:
                df[f'stat_{k}_change(t_1)_ratio{suffix}'] = data['Close'].pct_change().shift()

    if len(df) == 0:
        return df.copy(), False


    df = _add_arima_forecasting(df, interval, **params)
    df = _add_garch_forecasting(df, interval, **params)

    """
    PerformanceWarning: DataFrame is highly fragmented. 
    This is usually the result of calling `frame.insert` many times,
    which has poor performance.  Consider joining all columns at once using
    pd.concat(axis=1) instead. To get a de-fragmented frame,
    use `newframe = frame.copy()`
    """

    new_df_columns = list(df.columns)

    # Check if the any new columns are added
    if len(new_df_columns) > initial_length_of_columns:
        is_updated = True

    return df.copy(), is_updated
