import datetime

from src.Simulator.DataProviders import AllStocksPrices
from src.Simulator.AlphaSimulator.TradeHistoryAnalyzer import TradeHistoryAnalyzer
from src.Simulator.AlphaSimulator.SimulatorUtils import TradeHistoryHolder

from src.Simulator.AlphaSimulator import run_alpha_strategies

def simulate(**params):
    '''
    Simulate the investment

    This function performs a simulation of an investment based on the given parameters.
    First, it loads the stock prices and the trade history. Then, it runs the alpha strategies
    and finally, it simulates the investment and returns the reports and the summary.

    Args:
    params: dict
        The parameters for the simulation

    Returns:
    reports_df: pd.DataFrame
        The reports of the simulation

    summary: pd.DataFrame
        The summary of the simulation
    '''

    print (f"Starting the simulation at {datetime.datetime.now()}")

    stock_histories = AllStocksPrices(**params)
    stock_histories.load()
    trade_history_holder = TradeHistoryHolder(**params)

    trade_history_holder.load()

    run_alpha_strategies(stock_histories, trade_history_holder, **params)

    # Sort the trade history
    trade_history_holder.sort()
    trade_history_holder.save()

    trade_history_analyzer = TradeHistoryAnalyzer(trade_history_holder, stock_histories)
    
    # Simulate the investment
    reports_df, summary, _ = trade_history_analyzer.simulate_investment(**params)

    return reports_df, summary