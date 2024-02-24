import numpy as np
import pandas_ta as ta

def add_rsi_strategy(df_in, **params):

    df = df_in.copy()
    slippage_rate = params['slippage_rate']

    df['rsi'] = ta.rsi(df['Close'], length=14)
    # df['rsi'] = df['rsi'].shift()

    df['trade_opening_price'] = df['Open']

    df['signal'] = 0
    long_mask = ((df['rsi'] < 20) & 
                (df['stat_market_index_change(t_1)_ratio_1d'].abs() < 0.015))
    short_mask = ((df['rsi'] > 80)  & 
                (df['stat_market_index_change(t_1)_ratio_1d'].abs() < 0.015))

    df.loc[long_mask, 'signal'] = 1
    df.loc[short_mask, 'signal'] = -1

    df['signal'] = df['signal'].shift(1)

    return df