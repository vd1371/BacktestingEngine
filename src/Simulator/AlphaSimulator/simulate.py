import datetime

from src.Simulator.DataProviders import AllStocksPrices
from src.Simulator.AlphaSimulator.TradeHistoryAnalyzer import TradeHistoryAnalyzer
from src.Simulator.AlphaSimulator.SimulatorUtils import TradeHistoryHolder

from src.Simulator.AlphaSimulator import run_alpha_strategies

def simulate(**params):
    print (f"Starting the simulation at {datetime.datetime.now()}")

    stock_histories = AllStocksPrices(**params)
    stock_histories.load()
    trade_history_holder = TradeHistoryHolder(**params)

    trade_history_holder.load()

    run_alpha_strategies(stock_histories, trade_history_holder, **params)

    trade_history_holder.sort()
    trade_history_holder.save()

    trade_history_analyzer = TradeHistoryAnalyzer(trade_history_holder, stock_histories)
    reports_df, summary, _ = trade_history_analyzer.simulate_investment(**params)

    return reports_df, summary