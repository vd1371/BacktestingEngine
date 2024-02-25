import os
import pandas as pd
from tqdm import tqdm

from collections import deque

from .Trade import Trade
from src.utils import get_cache_key_for_params
from src.utils import convert_time_columns_to_datetime

class TradeHistoryHolder:

    '''
    The class to hold the trade history

    This class holds the trade history and provides methods to add, sort, convert to dict,
    convert to df, save, and load the trade history.

    If should_cache is True, it saves the trade history to the cache directory.
    If should_load_from_cache is True, it loads the trade history from the cache directory.
    The user will be asked if they want to continue if should_load_from_cache is True.

    You don't need to modify this class most of the times.
    '''

    def __init__(self, should_cache = True, **params):

        self.market = params["market"]
        self.reset()
        self.is_loaded = False
        self.should_cache = should_cache
        self.should_load_from_cache = params["should_load_from_cache"]
        self.enums = params['enums']
        if should_cache:
            self.cache_key = get_cache_key_for_params(params)
            self.direc = os.path.join(
                self.enums.POTENTIAL_TRADES_DIR,
                f"{self.cache_key}.csv"
                )
            # This is the file to be kept as the reference for research. The
            # research modules work based on this file to interpret the results
            self.direc_original = os.path.join(
                self.enums.POTENTIAL_TRADES_DIR,
                f"{self.cache_key}-Original.csv"
                )

    def reset(self):
        self.history = deque()
        self.trade_report_dfs = {}

    def reset_for_simulation(self):
        for trade in self.history:
            trade.reset_trade_for_simulation()

    def add(self, a_trade, market):
        self.history.append(a_trade)

    def add_many(self, list_of_trades, market):
        self.history += list_of_trades

    def sort(self):
        if self.is_loaded:
            return
        self.history = list(self.history)
        self.history.sort(key=lambda x: (x.opening_time))

    def convert_to_dict(self):
        history_dict = None
        if history_dict is None:
            history_dict = {}
        for trade in self.history:

            if history_dict.get(trade.symbol) is None:
                history_dict[trade.symbol] = []

            history_dict[trade.symbol].append(trade)

        return history_dict

    def convert_trade_history_to_df(self):
        
        holder = []
        for trade in self.history:
            holder.append(trade.get_dict())

        df = pd.DataFrame.from_dict(holder)
        report_df = df

        return report_df

    def save(self):
        if self.is_loaded or not self.should_cache:
            return
        
        df_trades = self.convert_trade_history_to_df()
        df_trades.to_csv(self.direc)
        df_trades.to_csv(self.direc_original)

    def load(self, df = None, force_load = False):
        if df is None and not self.should_cache:
            return

        self.reset()
        if df is None and not os.path.exists(self.direc):
            return
        
        if df is None and not self.should_load_from_cache:
            return

        if df is None and not force_load:
            answer = input(
                "should_load_from_cache is True, I'm loading from cache. " + \
                "Do you want me to continue(Y/n)? "
                )
            
            if answer != "Y":
                return
            
        if force_load:
            print ("I'm force loading the cache, this is dangerous, BE CAREFUL")

        try:

            if df is None:
                df = pd.read_csv(
                    self.direc,
                    index_col = 0
                    )
                
                df = convert_time_columns_to_datetime(df, self.market)
            
            for (i, row) in tqdm(list(df.iterrows())):
                row_dict = row.to_dict()
                trade = Trade(opening_time = row_dict['opening_time'])
                trade.load_from_dict(row_dict)
                self.add(trade, self.market)

            self.is_loaded = True
        except FileNotFoundError:
            print ("The potential trades could not be found")