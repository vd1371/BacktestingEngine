import warnings

def calculate_fees_and_taxes(
    invested_budget,
    capital_gain,
    num_shares,
    **params):

    fees = 0
    taxes = 0
    should_warn = params.get("should_log")
    market = params['market']

    if market.startswith("US"):
        # fees += calculate_robinhood_fee(invested_budget, num_shares)
        fees += min(max(0.005 * num_shares, 1), 0.01*invested_budget) # IBKR
        
        sec_transaction_fee = 0.000008 * invested_budget
        finra_transaction_fee = 0.000145 * num_shares

        return fees + sec_transaction_fee + finra_transaction_fee, taxes

    elif market == "HK":
        # fees += invested_budget * 0.27/100 # Webull
        fees += max(invested_budget * 0.08/100, 18) # IBKR
        fees += invested_budget * 0.26/100 # Stamp Duty
    
    elif market == "Crypto":
        fees += invested_budget * 0.2/100 # Binance

    else:
        if should_warn:
            warnings.warn("Fees and taxes for other markets are not correct")
        if capital_gain > 0:
            fees += invested_budget * 0.2/1000
    
    return fees, taxes