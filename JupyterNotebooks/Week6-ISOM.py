import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Download the ticker from yfinance
# import yfinance as yf

# def download_ticker(ticker):

#     cache_dir = os.path.join('Database', '_cache', f"{ticker}-1d-5.csv")

#     if os.path.exists(cache_dir):
#         return pd.read_csv(cache_dir, index_col=0)

#     data = yf.download(ticker, start="2016-01-01", end="2023-12-29")
#     data.to_csv(cache_dir)

#     return data


# ticker = ['AGG', 'QQQ', 'TMF', 'VOO', 'GLD', '^GSPC']

# for t in ticker:
#     print ("Downloading: ", t)
#     download_ticker(t)


# Load the data
def load_data(ticker):
    cache_dir = os.path.join('Database', '_cache', f"{ticker}-1d-5.csv")

    if os.path.exists(cache_dir):
        dir = cache_dir
    else:
        dir = os.path.join('Database', 'US', f"{ticker}-1d-5.csv")

    data = pd.read_csv(dir, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data.index = data.index.tz_convert('US/Eastern')

    return data

def save_data(ticker, df):
    cache_dir = os.path.join('Database', '_cache', f"{ticker}-1d-5.csv")
    df.to_csv(cache_dir)

tickers = ['AGG', 'QQQ', 'TMF', 'VOO', 'GLD', '^GSPC']

def find_goodness_of_fit(series1, series2):
    # Concat the series and drop nan
    df = pd.concat([series1, series2], axis=1)
    df = df.dropna()

    series1 = df.iloc[:, 0]
    series2 = df.iloc[:, 1]

    corr = np.corrcoef(series1, series2)[0, 1]
    mse = np.mean((series1 - series2) ** 2)
    mape = np.mean(np.abs((series1 - series2) / series1))

    return corr, mse, mape

window = 22
data_holder = {}
for ticker in tickers:
    data_holder[ticker] = load_data(ticker)

def get_facts(series, should_print=True):
    average = series.mean()
    std = series.std()
    sharpe = average / std * np.sqrt(252)

    if should_print:
        print ("Average: ", average)
        print ("Std: ", std)
        print ("Sharpe: ", sharpe)

    return average, std, sharpe

# # Calculate the daily returns
for ticker in tickers:
    data_holder[ticker]['daily_return'] = data_holder[ticker]['Close'].pct_change() * 100
    data_holder[ticker]['r_star'] = (data_holder[ticker]['daily_return'] - data_holder[ticker]['daily_return'].rolling(window=window).mean()) ** 2


# ---------------------------------------------------------------------------- #
#                        Historical Moving Average                             #
# ---------------------------------------------------------------------------- #
# # Calculate the historical
# for ticker in tickers:

#     data_holder[ticker]['h_vol_t'] = data_holder[ticker]['r_star'].rolling(window=window).mean()
#     data_holder[ticker]['h_vol_t_hat'] = data_holder[ticker]['h_vol_t'].shift(1)

#     mean_vol = data_holder[ticker]['h_vol_t'].mean()
#     data_holder[ticker]["h_vol_t_hat"] = data_holder[ticker]["h_vol_t_hat"].fillna(mean_vol)
#     data_holder[ticker]["r_star_hat"] = data_holder[ticker]["h_vol_t_hat"] * window - data_holder[ticker]["r_star"].shift().rolling(window=window-1).sum()

#     save_data(ticker, data_holder[ticker])

# # Find the goodness of fit for all and print as a table
# holder = []
# for ticker in tickers:
#     corr, mse, mape = find_goodness_of_fit(data_holder[ticker]['h_vol_t_hat'], data_holder[ticker]['r_star'])
#     holder.append([ticker, corr, mse, mape])

# df = pd.DataFrame(holder, columns=['Ticker', 'Correlation', 'MSE', 'MAPE'])
# print ("Based on Historical Data")
# print(df)

# # Plot the goodness of fit for CRM using scatter plot
# plt.scatter(data_holder['^GSPC']['r_star'], data_holder['^GSPC']['h_vol_t_hat'])
# plt.xlabel('Volatility')
# plt.ylabel('Volatility_hat')
# plt.title('Historical Volatility ^GSPC')
# plt.show()



# ---------------------------------------------------------------------------- #
#                      Exponentially Moving Average                            #
# ---------------------------------------------------------------------------- #
for ticker in tickers:
    data_holder[ticker]['ew_vol_t'] = data_holder[ticker]['r_star'].ewm(span=window).mean()
    data_holder[ticker]['ew_vol_t_hat'] = data_holder[ticker]['ew_vol_t'].shift(1)

    # data_holder[ticker]['vol_1'] = data_holder[ticker]['r_star'].ewm(span=66).mean()
    # data_holder[ticker]['vol_2'] = data_holder[ticker]['r_star'].ewm(span=11).mean()

    # data_holder[ticker]['vol_hat'] = \
    #     (data_holder[ticker]['vol_1'].shift(1) + \
    #         data_holder[ticker]['vol_2'].shift(1) + \
    #             data_holder[ticker]['ew_vol_t_hat']) / 3

    # Dropna
    data_holder[ticker] = data_holder[ticker].fillna(0)

    save_data(ticker, data_holder[ticker])

# Find the goodness of fit for all and print as a table
holder = []
for ticker in tickers:
    # corr, mse, mape = find_goodness_of_fit(data_holder[ticker]['ew_vol_t_hat'], data_holder[ticker]['r_star'])
    corr, mse, mape = find_goodness_of_fit(data_holder[ticker]['ew_vol_t_hat'], data_holder[ticker]['r_star'])
    holder.append([ticker, corr, mse, mape])

df = pd.DataFrame(holder, columns=['Ticker', 'Correlation', 'MSE', 'MAPE'])
print ("Based on Exponential Moving Average")
print(df)

# Plot the goodness of fit for CRM using scatter plot
# plt.scatter(data_holder['CRM']['r_star'], data_holder['CRM']['ew_vol_t_hat'])
# plt.xlabel('Volatility')
# plt.ylabel('Volatility_hat')
# plt.title('Exponential Moving Average Volatility CRM')
# plt.show()

# ---------------------------------------------------------------------------- #
#                      Fitting the risk metrics model                          #
# ---------------------------------------------------------------------------- #

# holder = []
# for ticker in tickers:
#     df = data_holder[ticker].copy()

#     df['daily_return'] = df['Close'].pct_change()
#     df['r_star'] = (df['daily_return'] - df['daily_return'].rolling(window=window).mean()) ** 2
#     df['vol'] = df['daily_return'].rolling(window=window).var()

#     lambda_ = 0.5
#     min_mse = 100000
#     best_lambda = 0
#     best_corr = 0
#     for lambda_ in range(1, 100):
#         lambda_ = lambda_ / 100
#         df['risk_metrics_vol'] = ((1 - lambda_) * df['daily_return'] ** 2 + lambda_ * df['vol'])
#         df['risk_metrics_vol'] = df['risk_metrics_vol'].shift().fillna(0)

#         # Find the goodness of fit for all and print as a table
#         df_for_calc = df.iloc[50:, :].copy()

#         corr, mse, mape = find_goodness_of_fit(df_for_calc['risk_metrics_vol'], df_for_calc['r_star'])

#         if mse < min_mse:
#             min_mse = mse
#             best_lambda = lambda_
#             best_corr = corr

#         # print (f"Risk Metrics Model with lambda = {lambda_}")
#         # print ("Correlation: ", corr)
#         # print ("MSE: ", mse)
#         # print ("MAPE: ", mape)

#     lambda_ = best_lambda
#     df['risk_metrics_vol'] = ((1 - lambda_) * df['daily_return'] ** 2 + lambda_ * df['vol'])
#     df['risk_metrics_vol'] = df['risk_metrics_vol'].shift().fillna(0)

#     # Find the goodness of fit for all and print as a table
#     df = df.iloc[50:, :].copy()

#     corr, mse, mape = find_goodness_of_fit(df['risk_metrics_vol'], df['r_star'])
#     holder.append([ticker, corr, mse, mape, lambda_])

# df = pd.DataFrame(holder, columns=['Ticker', 'Correlation', 'MSE', 'MAPE', 'Lambda'])
# print (df)

# ---------------------------------------------------------------------------- #
#                      GARCH Model                                             #
# ---------------------------------------------------------------------------- #

# for ticker in tickers:

#     if 'garch_vol_t_hat' in data_holder[ticker]:
#         continue

#     tmp_holder = [None for i in range(window)]

#     for i in range(window, len(data_holder[ticker])):

#         print (f"{i}/{len(data_holder[ticker])}", end='\r')

#         tmp = data_holder[ticker]['daily_return'].iloc[i-window:i]

#         model = arch_model(tmp, vol='Garch', p=1, q=1, rescale=False)
#         model_fit = model.fit(disp='off')

#         # Print the conditional volatility
#         vol_forecast = model_fit.forecast(horizon=1)
#         tmp_holder.append(vol_forecast.variance.values[-1][0])

#     data_holder[ticker]['garch_vol_t_hat'] = tmp_holder

#     # Dropna
#     data_holder[ticker] = data_holder[ticker].fillna(0)

#     save_data(ticker, data_holder[ticker])

# # Find the goodness of fit for all and print as a table
# holder = []
# for ticker in tickers:
#     corr, mse, mape = find_goodness_of_fit(data_holder[ticker]['garch_vol_t_hat'], data_holder[ticker]['r_star'])
#     holder.append([ticker, corr, mse, mape])

# df = pd.DataFrame(holder, columns=['Ticker', 'Correlation', 'MSE', 'MAPE'])
# print ("Based on GARCH")
# print(df)

# # Plot the goodness of fit for AAPL using scatter plot
# plt.scatter(data_holder['AAPL']['r_star'], data_holder['AAPL']['garch_vol_t_hat'])
# plt.xlabel('Volatility')
# plt.ylabel('Volatility_hat')
# plt.title('GARCH Volatility')
# plt.show()


# ---------------------------------------------------------------------------- #
#                      EGARCH Model                                             #
# ---------------------------------------------------------------------------- #

# for ticker in tickers:

#     if 'egarch_vol_t_hat' in data_holder[ticker]:
#         continue

#     tmp_holder = [None for i in range(window)]

#     for i in range(window, len(data_holder[ticker])):

#         print (f"{i}/{len(data_holder[ticker])}", end='\r')

#         tmp = data_holder[ticker]['daily_return'].iloc[i-window:i]

#         model = arch_model(tmp, vol='EGARCH', p=1, q=1, rescale=False)
#         model_fit = model.fit(disp='off')

#         # Print the conditional volatility
#         vol_forecast = model_fit.forecast(horizon=1)
#         tmp_holder.append(vol_forecast.variance.values[-1][0])

#     data_holder[ticker]['egarch_vol_t_hat'] = tmp_holder

#     # Dropna
#     data_holder[ticker] = data_holder[ticker].fillna(0)

#     save_data(ticker, data_holder[ticker])

# # Find the goodness of fit for all and print as a table
# holder = []
# for ticker in tickers:
#     corr, mse, mape = find_goodness_of_fit(data_holder[ticker]['egarch_vol_t_hat'], data_holder[ticker]['r_star'])
#     holder.append([ticker, corr, mse, mape])

# df = pd.DataFrame(holder, columns=['Ticker', 'Correlation', 'MSE', 'MAPE'])
# print ("Based on EGARCH")
# print(df)


# ---------------------------------------------------------------------------- #
#                      Strategy 1 - Based on Volatility                         #   
# ---------------------------------------------------------------------------- #


## Load vix and snp
vix = load_data('^VIX')
snp = load_data('^GSPC')


# Plot the Close price of VIX and S&P on two different axis
# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('Date')
# ax1.set_ylabel('VIX', color=color)
# ax1.plot(vix.index, vix['Close'], color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('S&P', color=color)  # we already handled the x-label with ax1
# ax2.plot(snp.index, snp['Close'], color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()


# Calculate the daily returns
vix['daily_return'] = vix['Close'].pct_change() * 100
snp['daily_return'] = snp['Close'].pct_change() * 100

# Find the sign of the daily returns
vix['sign'] = np.sign(vix['daily_return'])
snp['sign'] = np.sign(snp['daily_return'])

t = 2
vix['sign_sum'] = vix['sign'].rolling(window=t).sum().shift(1)
snp['sign_sum'] = snp['sign'].rolling(window=t).sum().shift(1)

# # Dropna
vix = vix.dropna()
snp = snp.dropna()

# # find the intersection indices
idx = vix.index.intersection(snp.index)

vix = vix.loc[idx]
snp = snp.loc[idx]

# # Signal return
vix['signal_return'] = 0
snp['signal_return'] = 0

# If sign_sim is 2 for both, short the S&P and long the VIX
# vix['signal_return'] = np.where((vix['sign_sum'] == t) & (snp['sign_sum'] == t), vix['daily_return'], vix['signal_return'])
# snp['signal_return'] = np.where((vix['sign_sum'] == t) & (snp['sign_sum'] == t), -snp['daily_return'], snp['signal_return'])

# # If sign_sim is -t for both, long the S&P and short the VIX
# vix['signal_return'] = np.where((vix['sign_sum'] == -t) & (snp['sign_sum'] == -t), -vix['daily_return'], vix['signal_return'])
# snp['signal_return'] = np.where((vix['sign_sum'] == -t) & (snp['sign_sum'] == -t), snp['daily_return'], snp['signal_return'])


# return_series = vix['signal_return'] + snp['signal_return']
# get_facts(return_series)


# print ("----------------------------------------------------")
# Check the idea of the cumreturn of the strategy

# best_sharpe = 0
# best_t = 0
# best_threshold = 0

# for t in range(3, 10):
#     for threshold in range(1, 10):
#         threshold = threshold / 100

#         # t = 5
#         vix['cum_return'] = vix['daily_return'].rolling(window=t).sum().shift()
#         snp['cum_return'] = snp['daily_return'].rolling(window=t).sum().shift()

#         # Fillna with 0
#         vix = vix.fillna(0)
#         snp = snp.fillna(0)

#         # Signal return
#         vix['signal_return'] = 0.0
#         snp['signal_return'] = 0.0

#         # threshold = 0.02
#         # If the sum of the last 5 days is above 5%, long the VIX and short the S&P
#         mask = (vix['cum_return'] + snp['cum_return'] > threshold)
#         vix.loc[mask, 'signal_return'] = vix.loc[mask, 'daily_return']
#         snp.loc[mask, 'signal_return'] = -snp.loc[mask, 'daily_return']

#         # If the sum of the last 5 days is below -5%, long the S&P and short the VIX
#         mask = (vix['cum_return'] + snp['cum_return'] < -threshold)
#         vix.loc[mask, 'signal_return'] = -vix.loc[mask, 'daily_return']
#         snp.loc[mask, 'signal_return'] = snp.loc[mask, 'daily_return']
                                        
#         return_series = vix['signal_return'] + snp['signal_return']
#         _, _, sharpe = get_facts(return_series, should_print = False)

#         if sharpe > best_sharpe:
#             best_sharpe = sharpe
#             best_threshold = threshold
#             best_t = t

# print ("Best Sharpe: ", best_sharpe)
# print ("Best t: ", best_t)
# print ("Best Threshold: ", best_threshold)

# Strategy 2 - Based on Volatility
holder = []
series_holder = {}
for ticker in tickers:

    df = data_holder[ticker].copy()
    df['daily_return'] = df['Close'].pct_change() * 100

    # # Calculate the historical volatility
    window = 22
    df['daily_changes'] = df['Close'] / df['Open'] - 1

    rolling_changes_std = df['daily_changes'].rolling(window=window).std()
    rolling_changes_std = rolling_changes_std.fillna(0)

    df['signal_return'] = 0.0

    # If daily_changes > 1 std, short the S&P
    short_mask = (df['daily_changes'] > rolling_changes_std).shift().fillna(False)
    df.loc[short_mask, 'signal_return'] = -1 * df.loc[short_mask, 'daily_return']

    # If daily_changes < -1 std, long the S&P
    long_mask = (df['daily_changes'] < -rolling_changes_std).shift().fillna(False)
    df.loc[long_mask, 'signal_return'] = df.loc[long_mask, 'daily_return']

    return_series_2 = df['signal_return']

    series_holder[ticker] = return_series_2

    average, std, sharpe = get_facts(return_series_2, should_print=False)

    holder.append([ticker, average, std, sharpe])

df = pd.DataFrame(holder, columns=['Ticker', 'Average', 'Std', 'Sharpe'])
print (df)

port = ['^GSPC', 'QQQ', 'TMF', 'VOO']

# Concat the series and drop nan
port_return = pd.concat([series_holder[ticker] for ticker in port], axis=1)

port_return.columns = port

port_return = port_return.dropna()
port_return_sum = port_return.sum(axis=1)


port_return_sum = port_return_sum / len(port)
get_facts(port_return_sum)


# print ("------------------------------------------------------------")
# portfolio = pd.concat([return_series, return_series_2], axis=1)
# portfolio.columns = ['Strategy 1', 'Strategy 2']
# portfolio_return = portfolio.sum(axis=1)

# get_facts(portfolio_return)