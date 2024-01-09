import pandas as pd
import pandas_ta as ta

def add_technical_indicators(df, interval = None, **params):

    ## TODO: ASSIGNMENT #3: Add some technical indicators here

    technical_indicators = {
        'ATR': 14,
    }

    suffix = "" if interval is None else f"_{interval}"

    initial_columns = list(df.columns)

    for indicator, period in technical_indicators.items():

        if indicator == 'ATR':
            if f'stat_ATR{suffix}' in df.columns:
                continue
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=period)
            df[f'stat_ATR{suffix}'] = atr.shift()

    """
    PerformanceWarning: DataFrame is highly fragmented. 
    This is usually the result of calling `frame.insert` many times,
    which has poor performance.  Consider joining all columns at once using
    pd.concat(axis=1) instead. To get a de-fragmented frame,
    use `newframe = frame.copy()`
    """

    updated_columns = list(df.columns)

    return df.copy(), len(updated_columns) > len(initial_columns)