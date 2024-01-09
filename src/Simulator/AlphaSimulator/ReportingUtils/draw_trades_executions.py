import os
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.utils import convert_time_columns_to_datetime

def draw_trades_executions(stock_histories, **params):

    market = params.get("market")
    stop_loss_percentage = params['stop_loss_percentage']
    take_profit_percentage = params['take_profit_percentage']
    enums = params['enums']

    n_to_be_plotted = params["n_executed_trade_for_plotting"]

    executed_trades = pd.read_csv(os.path.join(enums.TRADE_REPORTS_DIR, "ExecutedTrades.csv"), index_col= 0)

    executed_trades = convert_time_columns_to_datetime(executed_trades, market)
        
    for index, row in executed_trades.iterrows():

        if index >= n_to_be_plotted:
            break

        symbol = row['symbol']

        df = stock_histories.data[symbol]

        opening_time = row['opening_time']
        opening_price = row['opening_price']
        closing_time = row['closing_time']
        closing_time = row['closing_time']
        closing_price = row['closing_price']
        reason_for_closing = row['reason_for_closing']
        trade_direction = row['trade_direction']

        
        td = pd.Timedelta(days=10)

        start_daily = opening_time - td
        end_daily = closing_time + td

        df_selected = df[(df.index >= start_daily) & (df.index <= end_daily)].copy()
        df_selected['Timestamp'] = mdates.date2num(df_selected.index.to_pydatetime())

        df_selected['exact_opening_time'] = np.nan
        df_selected['closing_time'] = np.nan
        df_selected['closing_time'] = np.nan

        df_selected.loc[df_selected.index == opening_time, 'exact_opening_time'] = opening_price
        df_selected.loc[df_selected.index == closing_time, 'closing_time'] = closing_price

        df_selected['stop_loss'] = opening_price * (1- trade_direction * stop_loss_percentage/100)
        df_selected['take_profit'] = opening_price * (1 + trade_direction * take_profit_percentage/100)

        first_dot_color = "g" if ((closing_price-opening_price)*trade_direction) > 0 else "r"

        ap0 = [
            mpf.make_addplot(df_selected['exact_opening_time'], type='scatter', color='orange'),
            mpf.make_addplot(df_selected['closing_time'], type='scatter', color=first_dot_color),
            mpf.make_addplot(df_selected['stop_loss'], color='r'),
            mpf.make_addplot(df_selected['take_profit'], color='g'),
        ]

        fig, axlist = mpf.plot(
            df_selected,
            type = 'candle',
            addplot = ap0,
            returnfig=True
            )
        
        # add a new suptitle
        info = f"TradeDirection: {trade_direction}" + " | " + \
            reason_for_closing + " | " + \
            str(opening_time) + " | " + \
            symbol
        fig.suptitle(info, fontsize=10)

        plt.savefig(os.path.join(enums.EXECUTED_TRADES_DIR, f"{index}-{symbol.replace('/', ' ')}.png"))
        plt.clf()