import numpy as np
import pandas as pd

def convert_time_columns_to_datetime(df, market):

    for col in df.columns:
        if "_time" not in col:
            continue

        if df[col].dtype != "object":
            continue
        
        df[col] = pd.to_datetime(df[col], utc=True)
        before_correction = df[col].copy()
        if market.startswith("US"):
            df[col] = df[col].dt.tz_convert('US/Eastern')
        elif market == "HK":
            df[col] = df[col].dt.tz_convert('Asia/Hong_Kong')
        elif market == "Crypto":
            # It's already UTC
            pass
        
        elif market == "JAPAN":
            df[col] = df[col].dt.tz_convert('Asia/Tokyo')

        elif market == "LONDON":
            df[col] = df[col].dt.tz_convert('Europe/London')

        else:
            raise NotImplementedError("Implement")
        after_correct = df[col]

        # Check if the correction is correct by checking the date of all rows
        for before, after in zip(before_correction, after_correct):
            if pd.isna(before) and pd.isna(after):
                continue
            assert before.date() == after.date(), "The index is not converted correctly"
        
        
    return df