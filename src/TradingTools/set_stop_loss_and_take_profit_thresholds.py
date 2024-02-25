
def set_stop_loss_and_take_profit_thresholds(open_price, signal = 1, **params):

    '''
    Set the stop loss and take profit thresholds

    This function is used to set the stop loss and take profit thresholds.

    Args:
        open_price: float
            The open price of the stock
            
        signal: int
            The signal of the trade. 1 for long and -1 for short

        **params: dict
            The parameters for the simulation

    Returns:
        stop_loss_threshold: float
            The stop loss threshold

        take_profit_threshold: float
            The take profit threshold
    '''

    take_profit_percentage = params.get("take_profit_percentage")
    stop_loss_percentage = params.get("stop_loss_percentage")

    stop_loss_threshold = open_price * (1 - signal * stop_loss_percentage/100)
    take_profit_threshold = open_price * (1 + signal * take_profit_percentage/100)

    return stop_loss_threshold, take_profit_threshold