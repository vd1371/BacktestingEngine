
def close_trade_at(
        trade,
        t,
        closing_price,
        reason,
        slippage = None, 
        **params
        ):
    
    '''
    Close the trade at a given price

    This function is used to close the trade at a given price.

    Args:
        trade: Trade
            The trade to close

        t: datetime
            The time to close the trade

        closing_price: float
            The price to close the trade

        reason: str
            The reason for closing the trade

        slippage: float
            The slippage rate

        **params: dict
            The parameters for the simulation

    Returns:
        None
    '''

    if slippage is None:
        slippage_rate = params['slippage_rate']
    else:
        slippage_rate = slippage

    closing_price = round(closing_price * (1-trade.trade_direction*slippage_rate), 4)

    if not trade.is_closed:

        trade.set_closing_info(
            price = closing_price,
            time = t,
            reason = reason,
        )