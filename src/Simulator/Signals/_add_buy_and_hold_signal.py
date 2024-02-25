import numpy as np

def add_buy_and_hold_signal(df_in, **params):

    '''
    This function adds a buy and hold signal to the dataframe.
    The signal is 1 for long.

    Two new columns should be added to the dataframe:
    - signal: which gives the signal for the trade. 1 for long, -1 for short, 0 for no trade
    - trade_opening_price: which gives the price at which the trade is assumed to be opened

    Using the open price as the trade opening price is a good assumption because
    we are assuming that the trade is opened at the open price of the stock.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input dataframe
    params : dict
        Dictionary of parameters
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added signal and trade_opening_price columns
    '''

    df = df_in.copy()

    # The first day, we buy
    df['signal'] = 0
    df['signal'].iloc[0] = 1
    df['trade_opening_price'] = df['Open']

    return df