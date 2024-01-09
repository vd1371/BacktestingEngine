
def close_trade_for_stop_loss(trade, t, stop_loss_threshold, reason = None, **params):

    slippage_rate = params['slippage_rate']

    if reason is None:
        reason = "Hitting Stop Loss Threshold"

    closing_price = round(stop_loss_threshold * (1-trade.trade_direction*slippage_rate), 4)

    if not trade.is_closed:

        trade.set_closing_info(
            price = closing_price,
            time = t,
            reason = reason
        )