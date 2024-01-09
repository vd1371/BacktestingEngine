
class Trade:

    def __init__(self, **params):

        self.symbol = params.get("symbol", "Unknown")
        self.market = params.get("market", "US")
        
        self.opening_time = params.get("opening_time")
        self.exact_opening_time = params.get("exact_opening_time")
        self.opening_price = params.get("opening_price")
        self.trade_direction = params.get("trade_direction")

        self.id = f"{self.market}-{self.symbol}-" + \
            self.opening_time.strftime("%Y%m%d-%H%M%S")
    
        self.closing_time = params.get("closing_time")
        self.closing_price = params.get("closing_price")

        self.number_of_take_profit_touches = 0

        self.reason_for_closing = params.get("reason_for_closing")
        
        self.is_closed = params.get("is_closed", False)

        self.is_successful = params.get("is_successful")
        self.PnL_ratio = params.get("PnL_ratio")

        # The following attributes will be set during the simulate_investment
        self.fees = params.get("fees")
        self.taxes = params.get("taxes")
        
        self.n_shares = params.get("n_shares", 0)
        self.invested_budget = params.get("invested_budget", 0)
        
        self.n_shares = params.get("n_shares", 0)
        self.invested_budget = params.get("invested_budget", 0)


        self.is_analyzed = False
        self.gain = 0

    def set_closing_info(self, price, time, reason):
        
        self.closing_price = price
        self.closing_time = time
        self.reason_for_closing = reason
        self.is_closed = True
    
        self._is_sucsessful()
    
    def set_fees_and_taxes(self, fees, taxes):

        self.fees = fees
        self.taxes = taxes
        
        
    def _is_sucsessful(self):
        self.PnL_ratio = \
            self.trade_direction * \
                (self.closing_price - self.opening_price) / self.opening_price
            
        self.is_successful = int(self.PnL_ratio > 0)

    def set_attribute(self, attribute, value):
        setattr(self, attribute, value)

    def find_total_duration(self):
        self.duration = (self.closing_time - self.opening_time).days

    def get_dict(self):
        self.find_total_duration()
        return self.__dict__
    
    def load_from_dict(self, trade_dict):
        for k, v in trade_dict.items():
            setattr(self, k, v)

    def reset_trade_for_simulation(self):
        self.fees = None
        self.taxes = None
        
        self.n_shares = None
        self.invested_budget = None

        self.is_analyzed = False

    def __str__(self):
        return f"Trade {self.__dict__}"
