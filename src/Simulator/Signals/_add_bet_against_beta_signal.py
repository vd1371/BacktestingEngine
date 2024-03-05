import os
import numpy as np
import pandas as pd
from config import LONG, SHORT, DONOTHING

def add_bet_against_beta_signal(stock_histories, **params):

    '''
    This function adds the bet against beta signal to the stock_histories

    NOTE: It is not complete and should be used as a starting point for the implementation of the bet against beta signal
    
    Args:
        stock_histories: AllStocksPrices
            The stock histories to add the bet against beta signal to

    Returns:
        None
    '''

    first = True
    for symbol in stock_histories.data:

        try:
            beta_series = stock_histories.data[symbol]['stat_beta_66_1d']
        except KeyError:
            continue
        beta_series.name = symbol

        if first:
            betas_df = beta_series
            first = False
        else:
            betas_df = pd.merge(betas_df, beta_series.to_frame(), left_index=True, right_index=True, how='left')

    # Some useful info: the beta is for the yesterday close that is shifted by 1
    N = 10

    first_dates = []
    last_dates = []
    for year in betas_df.index.year.unique():
        for month in betas_df.index.month.unique():
            tmp_df = betas_df[(betas_df.index.year == year) & (betas_df.index.month == month)]
            if len(tmp_df) == 0:
                continue

            first_dates.append(tmp_df.index[0])
            last_dates.append(tmp_df.index[-1])


    # Add DONOTHING signal to all stocks
    for symbol in stock_histories.data:
        stock_histories.data[symbol]['signal'] = DONOTHING

    for date in first_dates:

        # Randomly select N stocks
        top_n_beta = betas_df.loc[date].sort_values(ascending=False).copy()
        top_n_beta.dropna(inplace=True)
        top_n_beta = list(top_n_beta.index)

        if len(top_n_beta) < 2*N:
            continue

        i = 0
        added_symbols = 0
        while i < 3*N and added_symbols < N:
            # stock_histories.data[symbol].loc[date, 'signal'] = SHORT
            symbol = top_n_beta[i]
            i += 1

            # if True:
            if (abs((stock_histories.data[symbol].loc[date, 'stat_beta_66_1d']) < 10)
                ):

                stock_histories.data[symbol].loc[date, 'signal'] = SHORT
                added_symbols += 1


        bottom_n_beta = betas_df.loc[date].sort_values(ascending=True)
        bottom_n_beta.dropna(inplace=True)
        bottom_n_beta = list(bottom_n_beta.index)

        i = 0
        added_symbols = 0
        while added_symbols < N:
            symbol = bottom_n_beta[i]
            i += 1

            if True:
                stock_histories.data[symbol].loc[date, 'signal'] = LONG
                added_symbols += 1


    for symbol in stock_histories.data:

        df = stock_histories.data[symbol]

        df['trade_opening_price'] = df['Open']
        
        df['risk_adjuster'] = (1 / df['stat_beta_66_1d'].abs())
        df['risk_adjuster'] = df['risk_adjuster'].fillna(1)
        df['risk_adjuster'] = df['risk_adjuster'].clip(upper=10)
        # df['risk_adjuster'] = 1

        # Adding the close signal
        last_dates = []
        for year in df.index.year.unique():
            for month in df.index.month.unique():
                tmp_df = df[(df.index.year == year) & (df.index.month == month)]
                if len(tmp_df) == 0:
                    continue

                last_dates.append(tmp_df.index[-1])


        df['close_long_signal'] = 0
        df['close_short_signal'] = 0

        # Add close signal flag to day_of_week_to_close
        df.loc[last_dates, 'close_long_signal'] = 1
        df.loc[last_dates, 'close_short_signal'] = 1

        stock_histories.data[symbol] = df.copy()
