import numpy as np

def add_random_signal(df_in, **params):

    df = df_in.copy()
    slippage_rate = params['slippage_rate']

    df['signal'] = np.random.choice([0, 1, -1], len(df), p=[0.4, 0.3, 0.3])
    df['trade_opening_price'] = df['Open'] * (1+ df['signal'] * slippage_rate)

    return df