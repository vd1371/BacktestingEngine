import datetime
import numpy as np
import pandas as pd

def convert_index_to_datetime(df, market, trading_interval = None):

    '''
    Convert the index to datetime
    the code converts the index to datetime based on the market and the trading interval
    This code works for the US, HK, and Japan markets

    If the market is US, the code converts the index to the US/Eastern timezone
    If the market is HK, the code converts the index to the Asia/Hong_Kong timezone
    If the market is Japan, the code converts the index to the Asia/Tokyo timezone
    If the market is London, the code converts the index to the Europe/London timezone

    If you want to add more markets, you can add them to the code based on the timezone of the market
    Add the new code following the same pattern as the existing code

    Args:
    df: pd.DataFrame
        The data
    market: str
        The market of the stock
    trading_interval: str
        The trading interval of the stock

    Returns:
        df.index: pd.DatetimeIndex
    '''

    df.index = pd.to_datetime(df.index, utc=True)
    before_correction = df.index.copy()
    if market.startswith("US"):
        df.index = df.index.tz_convert('US/Eastern')

        if trading_interval in ["1d", "1 day"]:
            df.index = df.index.map(lambda x: x.replace(hour=9, minute=30, second=0, microsecond=0))

    elif market == "HK":
        # Sometimes, the hour and minute are not 1 and 30, they are 16 and 0
        # So, we need to convert them to 1 and 30 which is the time when the
        # HK market opens
        if trading_interval in ["1d", "1 day"]:
            df.index = df.index.map(lambda x: x.replace(hour=1, minute=30, second=0, microsecond=0))
        df.index = df.index.tz_convert('Asia/Hong_Kong')

        if trading_interval in ["1d", "1 day"]:
            ## convert the index to the beginning of the day
            df.index = df.index.map(lambda x: x.replace(hour=9, minute=30, second=0, microsecond=0))

    elif market == "JAPAN":
        df.index = df.index.tz_convert('Asia/Tokyo')
        if trading_interval in ["1d", "1 day"]:
            df.index = df.index.map(lambda x: x.replace(hour=9, minute=0, second=0, microsecond=0))

    elif market == "LONDON":
        df.index = df.index.tz_convert('Europe/London')
        if trading_interval in ["1d", "1 day"]:
            df.index = df.index.map(lambda x: x.replace(hour=8, minute=0, second=0, microsecond=0))

    elif market == "Crypto":
        pass

    else:
        raise NotImplementedError(
            "Implement"
        )
    
    after_correction = df.index
    # Check if the correction is correct by checking the date of all rows

    if market not in ['JAPAN', 'LONDON']:
        for before, after in zip(before_correction, after_correction):
            if not before.date() == after.date():
                print (market, trading_interval, before.date(), after.date())
                breakpoint()
                raise ValueError(f"The index is not converted correctly: {before.date()} != {after.date()}")
        
    return df.index