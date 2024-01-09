import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def _add_arima_forecasting(df, interval, **params):

    col_name = "stat_yhat"
    should_add_arima_forecasting = params["should_add_arima_forecasting"]

    if not should_add_arima_forecasting or col_name in df.columns:
        return df
    
    ## TODO: Add ARIMA forecasting

    return df