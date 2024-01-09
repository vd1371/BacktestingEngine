import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import pandas_ta as ta

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from src.Simulator.DataProviders import AllStocksPrices

def analyze_raw_data(timeframe, **params):
    
    market = params['market']

    base_direc = os.path.join("reports", f"{market}", f"RawDataReport-{timeframe}")
    if not os.path.exists(base_direc):
        os.makedirs(base_direc)

    stock_histories = AllStocksPrices(**params)
    stock_histories.load()

    data = _check_params_get_data(timeframe, stock_histories, **params)

    _get_basic_statistics(data, base_direc, **params)
    _plot_heatmap(data, base_direc, **params)

    print ("analyze_raw_data done")


def _check_params_get_data(timeframe, stock_histories, **params):

    data = stock_histories.data_local
        
    for symbol in list(data.keys()):
        data[symbol] = data.pop(symbol)

    return data



def _get_basic_statistics(data, base_direc, **params):

    holder = {}
    for symbol, df in data.items():
        
        res = {}

        df['returns'] = df['Close'] / df['Open'] - 1
        res['returns_CoV'] = df['returns'].std() / df['returns'].mean()

        res['returns_autocorr'] = (df['Close'] / df['Open'] - 1).autocorr()

        res['total(%)'] = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

        res['O/L_mean'] = (df['Open'] / df['Low'] - 1).mean() * 100
        res['O/L_CoV'] = ((df['Open'] / df['Low'] - 1) * 100).std() / res['O/L_mean']

        res['H/L_mean'] = (df['High'] / df['Low'] - 1).mean() * 100
        res['H/L_CoV'] = ((df['High'] / df['Low'] - 1) * 100).std() / res['H/L_mean']

        res['O/C_abs_mean'] = (df['Open'] / df['Close'] - 1).abs().mean() * 100
        res['O/C_abs_CoV'] = ((df['Open'] / df['Close'] - 1) * 100).abs().std() / res['O/C_abs_mean']

        res['O_to_C_t_1_mean'] = (df['Open'] / df['Close'].shift() - 1).mean() * 100
        res['O_to_C_t_1_CoV'] = ((df['Open'] / df['Close'].shift() - 1) * 100).std() / res['O_to_C_t_1_mean']

        holder[symbol] = res

        df.drop(columns = ['returns'], inplace=True)

    df = pd.DataFrame(holder).T

    # Round all numbers to 2 decimal points
    df = df.round(2)

    df.to_csv(os.path.join(base_direc, "BasicStatistics.csv"))



def _plot_heatmap(data, base_direc, **params):

    # Plot for the close prices
    holder = []
    for symbol, df in data.items():
        holder.append(df['Close'])

    df = pd.concat(holder, axis=1)
    df.columns = data.keys()

    plt.figure(figsize=(10, 10))
    cmap = sns.diverging_palette(10, 120, as_cmap=True)
    sns.heatmap(df.corr(), cmap=cmap)

    df.to_csv(os.path.join(base_direc, "CorrelationOfClose.csv"))
    plt.title("Correlation of Close Prices")
    plt.savefig(os.path.join(base_direc, "Heatmap-CorrClose.png"))
    plt.clf()

    # Plot for daolu returns
    holder = []
    for symbol, df in data.items():
        holder.append(df['Close'] / df['Open'] - 1)

    df = pd.concat(holder, axis=1)
    df.columns = data.keys()
    cmap = sns.diverging_palette(10, 120, as_cmap=True)
    sns.heatmap(df.corr(), cmap=cmap)

    df.to_csv(os.path.join(base_direc, "CorrelationOfReturns.csv"))
    plt.title("Correlation of Returns")
    plt.savefig(os.path.join(base_direc, "Heatmap-CorrReturns.png"))
    plt.clf()


def _get_sharpe(returns):

    sharpe = returns.mean() / (returns.std() + 1e-6) * (252**0.5)

    return sharpe