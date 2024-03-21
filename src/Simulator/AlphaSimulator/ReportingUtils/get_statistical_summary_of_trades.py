import numpy as np
import pandas as pd
from scipy.stats import norm
from pandas.api.types import is_float_dtype
from datetime import datetime, timedelta

from config import LONG, SHORT

from src.utils import convert_time_columns_to_datetime

def get_statistical_summary_of_trades(trades_df, df_g = None, **params):

    '''
    ## GUIDE: Step 10

    Get the statistical summary of the trades

    This function gets the statistical summary of the trades.

    Args:
        trades_df: pd.DataFrame
            The trades history

        df_g: pd.DataFrame
            The daily budget dataframes

        params: dict
            The parameters for the simulation

    Returns:
        summaries_df: pd.DataFrame
    '''

    market = params['market']
    trades_df = convert_time_columns_to_datetime(trades_df, market)

    summaries = []
    summary = {}
    summary['Market'] = params['market']

    summary = add_statistical_reports(trades_df, summary, df_g, **params)
    # Combine the trading params and summary_of_marker
    summary = {**summary, **params}
    summaries.append(summary)

    # Split the df into equal pieces and report
    for period, period_df, period_df_g in _split_df_into_pieces(trades_df, df_g, **params):

        summary = {}
        summary['Market'] = params['market'] + f"-{period}"
        summary = add_statistical_reports(
            period_df,
            summary,
            period_df_g,
            is_chunk = True,
            **params
        )

        # Combine the trading params and summary_of_marker
        summary = {**summary, **params}
        summaries.append(summary)

    summaries_df = _convert_to_dataframe(summaries)
    summaries_df = summaries_df.drop(
        columns=list(params.keys()) + ['Approach'], errors='ignore'
        ).fillna("      ")
    
    return summaries_df

def add_statistical_reports(df, summary, df_g = None, is_chunk = False, **params):


    '''
    This function adds the statistical reports to the summary

    Args:
        df: pd.DataFrame
            The trades history

        summary: dict
            The summary of the trades

        df_g: pd.DataFrame
            The daily budget dataframes

        is_chunk: bool
            Whether the df is a chunk

        params: dict
            The parameters for the simulation

    Returns:
        summary: dict

    '''

    years = params.get("years_to_consider")
    summary['win'] = df['is_successful'].mean()
    summary['long_win'] = df[df["trade_direction"] == LONG]['is_successful'].mean()
    summary['short_win'] = df[df["trade_direction"] == SHORT]['is_successful'].mean()

    summary['Trades'] = len(df)
    summary['n_long'] = len(df[df["trade_direction"] == LONG])
    summary['n_short'] = len(df[df["trade_direction"] == SHORT])

    if df_g is not None:
        init_budget = df_g['PortfolioValue'].iloc[0]
        last_budget = df_g['PortfolioValue'].iloc[-1]
        summary['return(%)'] = (last_budget/init_budget-1) * 100

        if is_chunk:
            summary['annual(%)'] = None
        else:
            summary['annual(%)'] = \
                ((1+summary['return(%)']/100)**(1/years)-1) * 100
            
        returns = df_g['PortfolioValue'].pct_change() * 100
        returns.dropna(inplace = True)

        average_daily_return = returns.mean()
        std_daily_return = returns.std()

        volatility = std_daily_return * np.sqrt(252)
        summary['volatility(%)'] = volatility

        confidence_level = 0.95
        z_score = norm.ppf(confidence_level)
        VaR_percentage = - z_score * std_daily_return

        summary['VaR(%)'] = VaR_percentage

        summary['sharpe'] = average_daily_return / std_daily_return * np.sqrt(252)
        


    ## TODO: ASSIGNMENT #1
    

    return summary


def _split_df_into_pieces(df, df_g, **params):

    """
    This function takes a dataframe and splits it into smaller pieces based on
    the specified reporting period.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to be split
    reporting_period : str, optional
        The reporting period to use for splitting the dataframe.
        Must be one of 'Quarter', 'CalendarYear', 'Month', or 'Year'
    years_to_consider : int, optional
        The number of years to consider when reporting_period is 'Year'
        
    Returns
    -------
    generator
        A generator that yields a tuple of the form (period, df_chunk) where
        `period` is the reporting period of the chunk and `df_chunk`
        is the corresponding dataframe chunk.
        
    Raises
    ------
    ValueError
        If the reporting_period specified is not one of the acceptable
        reporting periods
    """

    reporting_period = params.get("reporting_period")
    acceptable_reporting_periods = ['Quarter', 'CalendarYear', 'Month', 'Year']
    if reporting_period not in acceptable_reporting_periods:
        raise ValueError(
            "reporting_period must be in " + f"{acceptable_reporting_periods}"
        )

    if reporting_period == "Year" or df_g is None:
        pieces = params.get("years_to_consider")
        start = df['opening_time'].min()

        if "closing_time" in df.columns:
            end = df['closing_time'].max()
        elif "closing_time_last_trade" in df.columns:
            end = df['closing_time_last_trade'].max()
        else:
            raise ValueError

        if df_g is not None:
            start = df_g.index.min()
            end = df_g.index.max()
        
        delta = end - start
        interval = delta.total_seconds() / pieces

        for i in range(pieces):
            
            chunk_start = start + timedelta(seconds=interval * i)
            chunk_end = min(start + timedelta(seconds=interval * (i+1)), end)
            
            df_chunk = df[
                (df['opening_time'] >= chunk_start) & 
                (df['opening_time'] <= chunk_end)]

            if df_g is not None:
                df_g_chunk = df_g[
                    (df_g.index >= chunk_start) & 
                    (df_g.index <= chunk_end)]
            else:
                df_g_chunk = None
            
            yield (f'{chunk_start.strftime("%d/%m/%y")}-{chunk_end.strftime("%d/%m/%y")}', df_chunk, df_g_chunk)

    else:
        # df['closing_time'] = pd.to_datetime(df['closing_time'], utc=True)
        # df['year'] = df['closing_time'].dt.year
        # df['fiscal_quarter'] = (df['closing_time'].dt.month - 1) // 3 + 1
        # df['month'] = df['closing_time'].dt.month


        df_g['year'] = df_g.index.year
        df_g['fiscal_quarter'] = (df_g.index.month - 1) // 3 + 1
        df_g['month'] = df_g.index.month

        if reporting_period == "Quarter":
            cols_to_be_grouped_by = ['year', 'fiscal_quarter']

        elif reporting_period == "CalendarYear":
            cols_to_be_grouped_by = ['year']
        
        elif reporting_period == "Month":
            cols_to_be_grouped_by = ['year', 'month']

        grouped = df_g.groupby(cols_to_be_grouped_by)
        split_df_g_s = {group: data for group, data in grouped}

        for period, period_df_g in split_df_g_s.items():
            
            period_df = df[
                (df['opening_time'] >= period_df_g.index.min()) & \
                (df['closing_time'] <= period_df_g.index.max())
            ]

            yield period, \
                period_df, \
                period_df_g.drop(
                    ['year', 'fiscal_quarter', 'month'],
                    axis=1,
                    errors='ignore'
                )


def _convert_to_dataframe(summary):

    df = pd.DataFrame.from_records(summary)
    for col in df.columns:
        if is_float_dtype(df[col].dtype):
            df[col] = df[col].round(2)

    if 'return(%)' in list(df.columns):
        df['return(%)'] = df['return(%)'].round(2)

        for col in ['annual(%)']:
            na_mask = df[col].notnull()
            df.loc[na_mask, col] = df.loc[na_mask, col].astype(float).round(1)
    
    return df

