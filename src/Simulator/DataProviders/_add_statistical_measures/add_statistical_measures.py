import pandas as pd
import numpy as np
import statsmodels.api as sm

from ._add_arima_forecasting import _add_arima_forecasting
from ._add_garch_forecasting import _add_garch_forecasting

def add_statistical_measures(
        df,
        macro_and_other_data = None,
        interval = None,
        **params):
    
    '''
    ## GUIDE: Step 3

    Add statistical measures to the data
    This function adds statistical measures to the data.
    Don't forget to shift the data before using it if you are using
    the high, low, or close prices and assume that the trade is opened at open.

    IMPORTANT NOTE: Add "stat_" prefix to the new columns that you add to the data.
    This is important because we use this prefix to get the statistical measures
    from the data in the simulator, signal generator, alpha strategiy, and research.

    Args:
        df: pd.DataFrame
            The data

        macro_and_other_data: dict

            It consists of information provided from the AllStockPrices class.
            It contains the dataframes of the other stocks, the market index.
            It is usually passed for the daily data, so that we can use the daily data of the other indices

        interval: str
            The interval of the data
    
    Returns:
        df: pd.DataFrame
    '''

    ## TODO: ASSIGNMENT #2: Add Beta and IV here

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

        for w in [22, 66]:
            if macro_and_other_data.get('market_index') is None:
                continue

            try:
                if f'stat_beta_{w}{suffix}' not in df:
                    merged_df = pd.merge(df['Close'], macro_and_other_data['market_index']['Close'], left_index=True, right_index=True, suffixes=('', '_market'))
                    merged_df['Close_return'] = merged_df['Close'].pct_change()
                    merged_df['Close_market_return'] = merged_df['Close_market'].pct_change()
                    merged_df = merged_df.dropna()

                    betas = [np.nan for i in range(w-1)]
                    idio_vols = [np.nan for i in range(w-1)]
                    for i in range(len(merged_df) - w + 1):
                        window_data = merged_df.iloc[i:i+w]
                        beta, idio_vol = calculate_beta(window_data['Close_return'], window_data['Close_market_return'])
                        betas.append(beta)
                        idio_vols.append(idio_vol)


                    betas = pd.Series(betas, index=merged_df.index)
                    betas = betas.shift()
                    betas.name = f'stat_beta_{w}{suffix}'

                    idio_vols = pd.Series(idio_vols, index=merged_df.index)
                    idio_vols = idio_vols.shift()
                    idio_vols.name = f"stat_IV_based_on_daily_return_market_{w}{suffix}"

                    # merge beta with df
                    df = df.join(betas)
                    df = df.join(idio_vols)

            except Exception as e:
                df[f'stat_beta_{w}{suffix}'] = np.nan
                df[f"stat_IV_based_on_daily_return_market_{w}{suffix}"] = np.nan

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


def calculate_beta(stock_returns, market_returns):
    # Calculate covariance and variance of returns
    # covariance = np.cov(stock_returns, market_returns)[0, 1]
    # market_variance = np.var(market_returns)

    # # Calculate beta
    # beta = covariance / market_variance

    # Regress stock_returns on market_returns using statsmodels
    model = sm.OLS(stock_returns, sm.add_constant(market_returns))
    results = model.fit()

    # Extract the beta
    beta = results.params.values[1]

    idio_vol = np.std(results.resid)

    return beta, idio_vol