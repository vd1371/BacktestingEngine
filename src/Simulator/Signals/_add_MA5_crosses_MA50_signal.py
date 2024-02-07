
def add_MA5_crosses_MA50_signal(df_in, **params):

    df = df_in.copy()

    # df['signal'] = np.random.choice([0, 1, -1], len(df), p=[0.4, 0.3, 0.3])
    # df['trade_opening_price'] = df['Open'] * (1+ df['signal'] * slippage_rate)

    df['ma_50'] = df['Close'].rolling(window=50).mean()
    df['ma_5'] = df['Close'].rolling(window=200).mean()

    # The moment the ma_5 crosses the ma_50 from below, we buy
    # The moment the ma_5 crosses the ma_50 from above, we sell

    df['signal'] = 0
    df['signal'] = (df['ma_5'] > df['ma_50']).astype(int)

    # We only need to trade when the signal changes
    df['signal'] = df['signal'].diff()

    # We need to shift the signal by one to avoid look-ahead bias
    df['signal'] = df['signal'].shift(1)

    df['trade_opening_price'] = df['Open']

    df.drop(['ma_50', 'ma_5'], axis=1, inplace=True)

    return df