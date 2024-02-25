import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from .SimulatorUtils import calculate_fees_and_taxes
from .SimulatorUtils import close_trade_at
from src.TradingTools import get_net_exposure
from config import LONG, SHORT

from src.utils import convert_index_to_datetime

import logging
simulation_logger = logging.getLogger("simulation_logger")

def simulate_investment(
        history_of_trades,
        stock_histories,
        **params
        ):
    
    ## GUIDE Step 8

    '''
    This function simulates the investment based on the given parameters.

    If you want to simulate the investment, you can use this function. It
    takes the history of trades and the stock histories and returns the
    reports and the summary.

    This allows you to simulate the investment based on the given parameters.

    Args:
        history_of_trades: list
            stock_histories: StockHistories

        **params: dict
            The parameters for the simulation

    Returns:
        df: pd.DataFrame
    '''
    
    if len(history_of_trades) == 0:
        raise ValueError(
            f"There seems to be something wrong with the {params['market']} market signals") 

    # Create a holder to keep the results of the simulation
    holder = SimulationResultsHolder(stock_histories, **params)

    # Get all the important dates
    dates = _get_all_important_dates(stock_histories)
    # dates_for_iteration = tqdm(dates) if params["should_log"] else dates
    dates_for_iteration = dates

    # Create a hash table of trades by time, to make it faster to access the trades by time
    trades_by_time = create_trades_by_time_hash_table(history_of_trades)

    next_valid_time = dates[0]
    for i, current_time in enumerate(dates_for_iteration):

        # If the current time is before the next valid time, we should skip this candle
        potential_trades = trades_by_time.get(current_time.replace(second=0, microsecond=0), None)
        
        candle_transactions = []
        for attempt in range(1):
            # This attempt to open position multiple times in the same candle
            # is for some special cases. Out of the context of this project.
            if potential_trades is None:
                break

            indices_of_executed_trades = []
            for i, trade in enumerate(potential_trades):

                # Check if the trade should be skipped, please see the function for more details
                if should_skip_this_trade(trade, stock_histories, holder, current_time, next_valid_time, **params):
                    continue
                
                # Open a new position, please see the function for more details
                trade, transaction_desc = \
                    open_a_new_position(
                        trade, i, indices_of_executed_trades, holder, **params)

                # If the trade is not executed, we should skip it
                if transaction_desc is not None:
                    candle_transactions.append(transaction_desc)

            if len(indices_of_executed_trades) == 0:
                break
            
            # Update the potential trades by removing the executed trades
            potential_trades = [potential_trades[i] for i in range(len(potential_trades)) if i not in indices_of_executed_trades]

        for i, trade in enumerate(holder.active_trades_queue):
            
            # If the trade is already closed, we should skip it, else
            # we should simulate the closure of the position
            if are_equal(current_time, trade.closing_time, **params) and \
                not trade.is_analyzed:

                _analyze_trade_in_simulation(trade, holder, candle_transactions, **params)


        # At the end of the candle, we should remove the completed trades
        holder.remove_completed_trades()
        locked_balance, net_exposure, \
            long_positions_value, short_positions_value = \
                _get_unrealized_value_of_trades(
                    holder.active_trades_queue,
                    stock_histories,
                    current_time,
                    **params
            )
        holder.update_net_exposure(net_exposure)

        holder.handle_end_of_candle_reporting(
            current_time, locked_balance, net_exposure,
            long_positions_value, short_positions_value,
            candle_transactions
        )

    # df_g is the report containing the daily portfolio value
    df_g = holder.get_daily_portfolio_value_report(dates)

    # df is the report containting the completed trades
    df = holder.get_completed_trades_report()
    
    return df, df_g

# ---------------------------------------------------------------------------- #
#                                                                              #
#                           Some useful functions                              #
#                                                                              #
# ---------------------------------------------------------------------------- #
def create_trades_by_time_hash_table(history_of_trades):
    '''
    This function creates a hash table of trades by time. The key is the
    opening time of the trade and the value is a list of trades that are
    opened at that time.

    The purpose of this function is to make it faster to access the trades
    by time.

    Args:
        history_of_trades: list
            The history of trades

    Returns:
        trades_by_time: dict
    '''
    trades_by_time = {}
    for trade in history_of_trades:
        opening_time = trade.opening_time.replace(second=0, microsecond=0)
        if opening_time not in trades_by_time:
            trades_by_time[opening_time] = []
        trades_by_time[opening_time].append(trade)
    return trades_by_time


def should_skip_this_trade(
        trade,
        stock_histories,
        holder,
        current_time,
        next_valid_time,
        **params
        ):
    '''
    This function checks if a trade should be skipped or not. It returns True
    if the trade should be skipped and False otherwise.

    The purpose of this function is to check if a trade should be skipped or not.
    There can be many reasons to skip a trade. For example, if the current time
    is before the next valid time, we should skip the trade. If the free balance
    is not enough to open a new position, we should skip the trade. If there is
    already an open position on the symbol of the trade, we should skip the trade.

    Args:
        trade: Trade
            The trade to check

        stock_histories: StockHistories
            The stock histories

        holder: SimulationResultsHolder
            The holder of the results of the simulation

        current_time: datetime
            The current time

        next_valid_time: datetime
            The next valid time

        **params: dict
            The parameters for the simulation

    Returns:
        bool
    '''

    if current_time < next_valid_time:
        return True

    if not holder.free_balance > 0:
        return True


def open_a_new_position(trade, index_of_trade, indices_of_executed_trades, holder, **params):

    '''
    This function opens a new position based on a given trade.

    The purpose of this function is to open a new position based on a given trade.
    It calculates the number of shares and the invested budget for the trade and
    then opens the trade if the invested budget is less than the free balance.

    Args:
        trade: Trade
            The trade to open

        index_of_trade: int
            The index of the trade in the list of trades

        indices_of_executed_trades: list
            The indices of the executed trades

        holder: SimulationResultsHolder
            The holder of the results of the simulation

        **params: dict
            The parameters for the simulation

    Returns:
        trade: Trade
    '''

    opening_price = trade.opening_price

    n_shares_for_trade, invested_budget_for_trade = \
        _calculate_shares_and_investment_considering_other_trades(
            opening_price = opening_price,
            budget = holder.budget,
            active_trades_queue = holder.active_trades_queue,
            **params
        )

    transaction_desc = None
    if invested_budget_for_trade < holder.free_balance and \
        holder.is_open_position_on_symbol[trade.symbol] is False:

        trade.n_shares = n_shares_for_trade
        trade.invested_budget = invested_budget_for_trade

        trade.n_shares= n_shares_for_trade
        trade.invested_budget= invested_budget_for_trade


        holder.add_trade_to_active_trades_queue(trade)
        indices_of_executed_trades.append(index_of_trade)
        holder.deduct_from_free_balance(trade.invested_budget)

        transaction_desc = f"{trade.id} opened. Invested: {invested_budget_for_trade}"

        if holder.is_open_position_on_symbol[trade.symbol] is True:
            raise ValueError("It should have been False")
        holder.update_is_open_position_on_symbol(trade.symbol, True)

    
    if trade.trade_direction == LONG:
        holder.add_to_long_positions_value(invested_budget_for_trade)
    elif trade.trade_direction == SHORT:
        holder.add_to_short_positions_value(invested_budget_for_trade)
    else:
        raise ValueError

    net_exposure = get_net_exposure(
        holder.long_positions_value,
        holder.short_positions_value
        )
    holder.update_net_exposure(net_exposure)
    
    return trade, transaction_desc


def are_equal(t1, t2, **params):
    if t1.date() == t2.date():
        return True
    return False

# Define a custom aggregation function to combine transactions
def combine_transactions(transactions):
    summary = ', '.join(transactions)
    summary = summary.replace("__exchange__ESXXXX__cfi", "")
    summary = summary.replace("__ticker", "")
    return summary


def _mark_trade_analyzed_set_taxes_fess_return_gain(
    trade,
    **params):

    '''
    The purpose of this function is to mark a trade as analyzed, set the taxes
    and fees, and return the gain of the trade.

    Args:
        trade: Trade
            The trade to analyze

        **params: dict
            The parameters for the simulation

    Returns:
        gain: float
    '''

    closing_price = trade.closing_price
    n_shares = trade.n_shares
    partial_invested_budget = trade.invested_budget

    capital_gain = n_shares * (closing_price - trade.opening_price) * trade.trade_direction
    
    fees, taxes = calculate_fees_and_taxes(
        invested_budget = partial_invested_budget,
        capital_gain = capital_gain,
        num_shares = n_shares,
        **params
    )

    trade.is_analyzed = True
    trade.fees = fees
    trade.taxes = taxes

    gain = round((capital_gain - fees - taxes), 4)
    trade.gain += gain

    return gain

def _get_unrealized_value_of_trades(
        active_trades_queue,
        stock_histories,
        current_time,
        **params):

    if len(active_trades_queue) == 0:
        return 0, 0, 0, 0

    locked_balance = 0
    net_exposure = 0
    long_positions_value = 0
    short_positions_value = 0

    for trade in active_trades_queue:
        symbol = trade.symbol
        opening_price = trade.opening_price
        trade_direction = trade.trade_direction

        df = stock_histories.data[symbol]

        try:
            price_at_current_time = df.loc[current_time, 'Close']
        except KeyError as e:
            price_at_current_time = df[df.index <= current_time]['Close'].iloc[-1]

        if not trade.is_analyzed:
            locked_balance_of_trade = trade.invested_budget + trade_direction*(price_at_current_time-opening_price)*trade.n_shares
            locked_balance += locked_balance_of_trade

            if trade_direction == LONG: long_positions_value += locked_balance_of_trade
            if trade_direction == SHORT: short_positions_value += locked_balance_of_trade

    net_exposure = get_net_exposure(long_positions_value, short_positions_value)

    return locked_balance, net_exposure, long_positions_value, short_positions_value


def _get_all_important_dates(stock_histories):

    '''
    This function gets all the important dates from the stock histories.

    The purpose of this function is to get all the important dates from the
    stock histories.

    Args:
        stock_histories: StockHistories
            The stock histories

    Returns:
        date_ranges: list
    '''

    tmp_holder = []
    for symbol, df in stock_histories.data.items():
        tmp_holder.extend(df.index.tolist())
    date_ranges = pd.Series(tmp_holder)
    date_ranges = date_ranges.drop_duplicates().sort_values()
    return date_ranges.tolist()


def _calculate_shares_and_investment_considering_other_trades(
    opening_price,
    budget,
    active_trades_queue = None,
    **params
    ):

    risk_level_percentage = params["risk_level_percentage"]
    stop_loss_percentage = params["stop_loss_percentage"]

    if risk_level_percentage > stop_loss_percentage:
        raise ValueError(
            "risk_level_percentage Must be smaller than stop_loss_percentage"
        )
    
    money_to_invest_based_on_budget = \
        risk_level_percentage/stop_loss_percentage * budget

    if active_trades_queue is None or len(active_trades_queue) == 0:
        min_of_invested_budget_in_other_trades = 0
    else:
        min_of_invested_budget_in_other_trades = \
            min(trade.invested_budget for trade in active_trades_queue)
        
    money_to_invest = max(
        money_to_invest_based_on_budget,
        min_of_invested_budget_in_other_trades
    )
    
    n_shares_to_invest = money_to_invest/opening_price
    money_to_invest = n_shares_to_invest * opening_price

    return n_shares_to_invest, round(money_to_invest, 4)


class SimulationResultsHolder:

    def __init__(self, stock_histories, **params):
        
        self.active_trades_queue = []
        self.completed_trades = []
        self.budget_after_each_trade = []
        self.budget_after_which_trade = []

        self.portfolio_values_list = []
        self.transactions_in_candle = []
        self.open_positions_at_the_end = []

        self.net_exposures_list = []
        self.long_positions_values_list = []
        self.short_positions_values_list = []

        init_budget = 1000000
        self.free_balance = init_budget
        self.budget = init_budget

        self.is_open_position_on_symbol = {}
        for symbol in stock_histories.data:
            self.is_open_position_on_symbol[symbol] = False

        # net_exposure, long_positions_value, short_positions_value
        self.net_exposure = 0
        self.long_positions_value = 0
        self.short_positions_value = 0

        self.market = params['market']
        self.trading_interval = params['trading_interval']

        self.portfolio_value_at_close_yeasterday = init_budget

    def add_trade_to_active_trades_queue(self, trade):
        self.active_trades_queue.append(trade)

    def remove_completed_trades(self):
        self.active_trades_queue = \
            _remove_completed_trades_from_trades_queue(self.active_trades_queue)

    def append_completed_trades(self, trade):
        self.completed_trades.append(trade)

    def append_budget_after_each_trade(self, budget):
        self.budget_after_each_trade.append(budget)

    def append_budget_after_which_trade(self, trade_id):
        self.budget_after_which_trade.append(trade_id)

    def append_portfolio_value(self, portfolio_values_list):
        self.portfolio_values_list.append(portfolio_values_list)

    def append_net_exposure_list(self, net_exposure):
        self.net_exposures_list.append(net_exposure)

    def append_long_positions_values_list(self, long_positions_value):
        self.long_positions_values_list.append(long_positions_value)

    def append_short_positions_values_list(self, short_positions_value):
        self.short_positions_values_list.append(short_positions_value)

    def update_net_exposure(self, net_exposure):
        self.net_exposure = net_exposure

    def handle_closure_of_a_part_of_a_trade(self, released_balance, gain):
        self.add_to_free_balance(round(released_balance + gain, 4))
        self.add_to_budget(gain)

    def add_to_free_balance(self, new_value):
        self.free_balance += new_value

    def deduct_from_free_balance(self, new_value):
        self.free_balance -= new_value

    def add_to_budget(self, new_value):
        self.budget += new_value

    def add_to_long_positions_value(self, new_value):
        self.long_positions_value += new_value
    
    def add_to_short_positions_value(self, new_value):
        self.short_positions_value += new_value

    def deduct_from_long_positions_value(self, new_value):
        self.long_positions_value -= new_value

    def deduct_from_short_positions_value(self, new_value):
        self.short_positions_value -= new_value

    def set_net_exposure(self, new_value):
        self.net_exposure = new_value

    def set_long_positions_value(self, new_value):
        self.long_positions_value = new_value

    def set_short_positions_value(self, new_value):
        self.short_positions_value = new_value

    def update_is_open_position_on_symbol(self, symbol, new_value):
        self.is_open_position_on_symbol[symbol] = new_value

    def handle_completion_of_a_trade(self, trade):
        self.append_budget_after_each_trade(self.budget)
        self.append_budget_after_which_trade(trade.id)
        self.append_completed_trades(trade)

        if self.is_open_position_on_symbol[trade.symbol] is False:
            raise ValueError("It should have been True")
        self.update_is_open_position_on_symbol(trade.symbol, False)

    def handle_end_of_candle_reporting(self,
            current_time, locked_balance, net_exposure,
            long_positions_value, short_positions_value,
            candle_transactions
            ):
        
        self.append_portfolio_value((current_time, self.free_balance + locked_balance))
        self.append_net_exposure_list(net_exposure)
        self.append_long_positions_values_list(long_positions_value)
        self.append_short_positions_values_list(short_positions_value)
        
        self.open_positions_at_the_end.append(" | ".join([trade.id for trade in self.active_trades_queue]))
        self.transactions_in_candle.append(combine_transactions(candle_transactions))

        if self.market.startswith("US") or self.market.startswith("HK"):
            self.portfolio_value_at_close_yeasterday = self.free_balance + locked_balance

        elif self.market in ["Crypto"]:
            self.portfolio_value_at_close_yeasterday = self.free_balance + locked_balance

        else:
            raise NotImplementedError(f"This market {self.market} is not implemented yet")

        
    def get_daily_portfolio_value_report(self, dates):

        df_g = pd.DataFrame()
        df_g['Date'] = dates
        df_g['PortfolioValue'] = [val[1] for val in self.portfolio_values_list]
        df_g['Transactions'] = self.transactions_in_candle
        df_g['open_positions_at_the_end'] = self.open_positions_at_the_end
        df_g['long_positions_values'] = self.long_positions_values_list
        df_g['short_positions_values'] = self.short_positions_values_list
        df_g.set_index("Date", inplace = True, drop=True)

        df_g.index = convert_index_to_datetime(df_g, self.market, self.trading_interval)

        # Group by date and aggregate the columns
        df_g['Date_tmp'] = df_g.index
        df_g = df_g.groupby(df_g.index.date).agg({
            'Date_tmp': 'first',
            'PortfolioValue': 'last',
            'Transactions': combine_transactions,
            'open_positions_at_the_end': 'last',
            'long_positions_values': 'last',
            'short_positions_values': 'last'
        })
        df_g.set_index('Date_tmp', inplace=True, drop=True)
        df_g = df_g.rename_axis('Date')
        
        return df_g
    
    def get_completed_trades_report(self):

        df = pd.DataFrame.from_dict(
            [trade.get_dict() for trade in self.completed_trades]
        )
        df['budget'] = self.budget_after_each_trade
        df['budget_after_which_trade'] = self.budget_after_which_trade

        df.sort_values(by = ['closing_time'], inplace=True)
        df.reset_index(inplace= True, drop=True)

        return df
    
    def print_active_trades_queue(self):
        if len(self.active_trades_queue) == 0:
            simulation_logger.info("There are no active trades remaining.")
            return
        for trade in self.active_trades_queue:
            simulation_logger.warning(
                f"{trade.id} {trade.is_analyzed} \
                    {trade.reason_for_closing} \
                        {trade.closing_time}"
            )
        simulation_logger.warning("These are the active trades remaining. Something should be wrong.")

def _remove_completed_trades_from_trades_queue(active_trades_queue):

    temporary_holder = []
    for trade in active_trades_queue:
        if not trade.is_analyzed:
            temporary_holder.append(trade)

    return temporary_holder

def close_all_active_trades(current_time, holder, stock_histories, message, candle_transactions, **params):
    
    # close_trade_at(trade, t_monitoring, close_price, "Max holding days touched", **params)

    for trade in holder.active_trades_queue:
        closing_price = stock_histories.data[trade.symbol].loc[current_time, 'Close']

        # The is_trade_closed flag is set to True in the run_alpha_strategy.
        # If we want to close the trade here at the close of the candle, we
        # need to set it to False again. Otherwise, the trade will not be closed.
        if not trade.is_analyzed:
            trade.is_closed = False
            # Now, we can close the trade at the close of the candle
            close_trade_at(trade, current_time, closing_price, message, **params)

        if not trade.is_analyzed:
            _analyze_trade_in_simulation(trade, holder, candle_transactions, **params)


    # next_valid_time will be next day 9:30 AM
    next_valid_time = current_time.replace(hour=9, minute=30) + timedelta(days=1)
    return next_valid_time


def _analyze_trade_in_simulation(trade, holder, candle_transactions, **params):
    
    '''
    This function analyzes a trade in the simulation.

    The purpose of this function is to analyze a trade in the simulation.
    It calculates the gain of the trade and then handles the closure of
    the trade.

    Args:
        trade: Trade
            The trade to analyze

        holder: SimulationResultsHolder
            The holder of the results of the simulation

        candle_transactions: list
            The transactions in the candle

        **params: dict
            The parameters for the simulation

    Returns:
        None
    '''

    gain = _mark_trade_analyzed_set_taxes_fess_return_gain(trade, **params)
    holder.handle_closure_of_a_part_of_a_trade(trade.invested_budget, gain)

    candle_transactions.append(f"{trade.id} closed.")
    holder.handle_completion_of_a_trade(trade)
