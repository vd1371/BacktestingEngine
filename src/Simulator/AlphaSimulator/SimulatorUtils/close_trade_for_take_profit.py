
def close_trade_for_take_profit(
    trade,
    t,
    stop_loss_threshold,
    take_profit_threshold,
    reason = None,
    **params
    ):

    '''
    Close the trade for take profit

    This function is used to close the trade for take profit.

    Args:
        trade: Trade
            The trade to close for take profit

        t: datetime
            The time to close the trade

        stop_loss_threshold: float
            The stop loss threshold

        take_profit_threshold: float
            The take profit threshold

        reason: str
            The reason for closing the trade

        **params: dict
            The parameters for the simulation

    Returns:
        stop_loss_threshold: float
    '''

    slippage_rate = params['slippage_rate'] 

    if reason is None:
        reason = "Touching Take Profit Threshold"

    closing_price = round(take_profit_threshold * (1-trade.trade_direction*slippage_rate), 4)

    trade.number_of_take_profit_touches += 1
    
    if not trade.is_closed:
        trade.set_closing_info(
            price = closing_price,
            time = t,
            reason = reason,
        )
    
    return stop_loss_threshold, take_profit_threshold