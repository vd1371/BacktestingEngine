import os
import datetime
import pytz

# decimal; ref: https://www.investopedia.com/articles/personal-finance/101515/comparing-longterm-vs-shortterm-capital-gain-tax-rates.asp
SHORT_TERM_GAIN_TAX_PORTION = 0.32

# decimal; ref: https://www.investopedia.com/articles/personal-finance/101515/comparing-longterm-vs-shortterm-capital-gain-tax-rates.asp
LONG_TERM_GAIN_TAX_PORTION = 0.15  

# DECIMAL; ONLY FOR $500+ sells; ref: https://cdn.robinhood.com/assets/robinhood/legal/RHF%20Fee%20Schedule.pdf
ROBINHOOD_REGULATORY_TRANSACTION_FEE = 0.0000229

# dollar; ref: https://cdn.robinhood.com/assets/robinhood/legal/RHF%20Fee%20Schedule.pdf
MINIMUM_REGYLATORY_TRANSACTION_AMOUNT = 500

# dollar per share; capped at $7.27 ref: https://cdn.robinhood.com/assets/robinhood/legal/RHF%20Fee%20Schedule.pdf
ROBINHOOD_TRADING_ACTIVITY_FEE = 0.000145

# dollar; ref: https://cdn.robinhood.com/assets/robinhood/legal/RHF%20Fee%20Schedule.pdf
TRADING_ACTIVITY_FEE_CAP = 7.27

# Slippage rate: We will not be able to make the trade at the price we see, we need an assumption

APPROXIMATE_TAX = 0.33

BINANCE_TRADING_FEE = 0.001

LONG = 1
SHORT = -1
DONOTHING = 0

class ENUMS:

    def __init__(self,
            for_test=False,
            **params) -> None:

        test_suffix = "Test" if for_test else ""

        self.TRADE_REPORTS_DIR = os.path.join(
            f"reports{test_suffix}",
            f"{params['market']}",
            f"{params['strategy_name']}"
            )
        if not os.path.exists(self.TRADE_REPORTS_DIR):
            os.makedirs(self.TRADE_REPORTS_DIR)

        # This is for saving the stat figures
        self.STAT_FIGURES_DIR = os.path.join(self.TRADE_REPORTS_DIR, "StatReports")
        if not os.path.exists(self.STAT_FIGURES_DIR):
            os.makedirs(self.STAT_FIGURES_DIR)


        # PotentilaTrades dir is used to save the trades as the results of run_alpha_strategy
        # And of course loading from it
        self.POTENTIAL_TRADES_DIR = os.path.join(self.TRADE_REPORTS_DIR, "PotentialTrades")
        if not os.path.exists(self.POTENTIAL_TRADES_DIR):
            os.makedirs(self.POTENTIAL_TRADES_DIR)

        # This is for the optimization results
        self.OPTIMIZATION_RESULTS_DIR = os.path.join(self.TRADE_REPORTS_DIR, "OptimizationResults")
        if not os.path.exists(self.OPTIMIZATION_RESULTS_DIR):
            os.makedirs(self.OPTIMIZATION_RESULTS_DIR)

        # This is for the executed trades
        self.EXECUTED_TRADES_DIR = os.path.join(self.TRADE_REPORTS_DIR, "ExecutedTradesPlots")
        if not os.path.exists(self.EXECUTED_TRADES_DIR):
            os.makedirs(self.EXECUTED_TRADES_DIR)

            
        self.DATABASE_CACHE_DIR = os.path.join("Database", "_cache")


def get_today(market = None) -> str:
    '''
    if market is US or US-IBKR, then will return the date in US/Eastern timezone 
    '''
    if market in ['US', 'US_IBKR']:
        utc_now = datetime.datetime.utcnow()
        eastern_tz = pytz.timezone('US/Eastern')
        eastern_now = utc_now.replace(tzinfo=pytz.utc).astimezone(eastern_tz)
        return eastern_now.strftime("%Y%m%d")

    return datetime.datetime.today().strftime("%Y%m%d")