
def set_stop_loss_and_take_profit_thresholds(open_price, signal = 1, **params):

    take_profit_percentage = params.get("take_profit_percentage")
    stop_loss_percentage = params.get("stop_loss_percentage")

    stop_loss_threshold = open_price * (1 - signal * stop_loss_percentage/100)
    take_profit_threshold = open_price * (1 + signal * take_profit_percentage/100)

    return stop_loss_threshold, take_profit_threshold