from .generate_report_for_trades_history import generate_report_for_trades_history
from .simulate_investment import simulate_investment

class TradeHistoryAnalyzer:

    def __init__(self, trade_history_holder, stock_histories):
        self.trade_history_holder = trade_history_holder
        self.stock_histories = stock_histories

    def simulate_investment(self, **params):
        
        # Simulate the investment
        report_dfs, daily_budget_dfs = \
            simulate_investment(
                self.trade_history_holder.history,
                self.stock_histories,
                **params
        )

        # Generate the report for the trades history
        summary_df = \
            generate_report_for_trades_history(
                report_dfs,
                daily_budget_dfs,
                self.stock_histories,
                **params)
        
        return report_dfs, summary_df, daily_budget_dfs