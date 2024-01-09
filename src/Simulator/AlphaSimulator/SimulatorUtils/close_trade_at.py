
def close_trade_at(
        trade,
        t,
        closing_price,
        reason,
        slippage = None, 
        **params
        ):

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