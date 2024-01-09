
def get_net_exposure(long_positions_value, short_positions_value):

    if isinstance(long_positions_value, (float, int)) and \
        long_positions_value + short_positions_value == 0:
        return 0

    try:
        return (long_positions_value - short_positions_value) / \
                (long_positions_value + short_positions_value) * 100
    except ZeroDivisionError:
        return 0