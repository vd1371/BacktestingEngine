import os
from copy import deepcopy

from .get_all_symbols import get_all_symbols

from ._data_loaders import load_data_from_yahoo_finance
from ._data_loaders import save_data_to_cache

from ._add_statistical_measures import add_statistical_measures
from ._add_technical_indicators import add_technical_indicators

import logging
simulation_logger = logging.getLogger("simulation_logger")

class AllStocksPrices:

    '''
    ## GUIDE: Step 2
    
    This class is responsible for loading the stock prices and the market index.
    It also adds statistical and technical indicators to the stock prices.
    '''

    def __init__(self, **params):
        '''
        Initialize the AllStocksPrices class

        Args:
        params: dict
            The parameters for the AllStocksPrices class
        '''

        for k, v in params.items():
            setattr(self, k, v)

        self.index = 0

        self.data = {}
        self.market_index = None

        # Cache directory for the stock prices
        self.cache_dir = _clear_and_warm_up_cache(self.market)


    def load(self):

        if self.should_log:
            print ("Warming up the AllStocksPrices with historical prices ...")

        market = self.market
        # Load the market index
        params_to_pass = deepcopy(self.__dict__)
        params_to_pass.update(trading_interval = "1d")

        if market.startswith("US"):
            # Load the market index
            self.market_index = load_data_from_yahoo_finance("^GSPC", market_macro_index = True, **params_to_pass)
            
            # Load the VIX index
            self.vix = load_data_from_yahoo_finance("^VIX", market_macro_index = True, **params_to_pass)
            
            # Load the gold futures
            self.gold_futures = load_data_from_yahoo_finance("GC=F", market_macro_index = True, **params_to_pass)
            
            # Load the 20 years treasuries
            self.treasuries_20years = load_data_from_yahoo_finance("TLT", market_macro_index = True, **params_to_pass)
            
            # Load the russel 2000
            self.russel_2000 = load_data_from_yahoo_finance("^RUT", market_macro_index = True, **params_to_pass)

            macro_and_other_data = {
                'market_index': self.market_index,
                'vix': self.vix,
                'gold_futures': self.gold_futures,
                'treasuries_20years': self.treasuries_20years,
                'russel_2000': self.russel_2000
            }

        elif market.startswith("HK"):
            self.market_index = load_data_from_yahoo_finance("^HSI", market_macro_index = True, **params_to_pass)
            macro_and_other_data = {
                'market_index': self.market_index
            }
        
        # Get all the symbols for the market
        symbols = get_all_symbols(**self.__dict__)
        
        # -------------------------------------------------------------------- #
        #                      YAHOO FINANCE DATA PROVIDER                     #
        # -------------------------------------------------------------------- #
        for symbol in symbols:
            df = load_data_from_yahoo_finance(symbol, **self.__dict__)
            self.data[symbol] = df
            
        # -------------------------------------------------------------------- #
        #            ADDING STATISCTICAL AND TECHNICAL INDICATORS              #
        # -------------------------------------------------------------------- #
        if len (self.data) > 0:
            params_to_pass = deepcopy(self.__dict__)
            params_to_pass.update(trading_interval = "1d")

            if self.should_log:
                print ("Adding statistical and technical indicators to daily data ...")

            for symbol, df_1d in self.data.items():

                df_1d, is_updated1 = add_statistical_measures(df_1d, macro_and_other_data=macro_and_other_data, interval = "1d", **params_to_pass)
                df_1d, is_updated2 = add_technical_indicators(df_1d, macro_and_other_data=macro_and_other_data, interval = "1d", **params_to_pass)
                self.data[symbol] = df_1d
                
                if any([is_updated1, is_updated2]):
                    save_data_to_cache(df_1d, symbol, **params_to_pass)

        
            
    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx] = val

    def get(self, symbol):

        if self.index >= len(self.data):
            self.index = 0

        results = {'symbol': symbol}

        if self.data.get(symbol, None) is not None:
            results['data'] = self.data[symbol]
        else:
            results['data'] = None

        if "US" in self.market:
            results['market_index'] = self.market_index
            results['vix'] = self.vix
            results['gold_futures'] = self.gold_futures
            results['treasuries_20years'] = self.treasuries_20years
            results['russel_2000'] = self.russel_2000

        results['cache_dir'] = self.cache_dir

        self.index += 1
        return results

def _clear_and_warm_up_cache(market):

    cache_dir = os.path.join("Database", market)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir