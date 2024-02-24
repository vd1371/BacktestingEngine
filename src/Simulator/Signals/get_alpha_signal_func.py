from ._add_random_signal import add_random_signal
from ._add_rsi_strategy import add_rsi_strategy

def get_alpha_signal_func(**params):

    strategy_name = params["strategy_name"]

    strategy_signal_func_dict = {
        "random": add_random_signal,
        'rsi': add_rsi_strategy
    }

    signal_func = strategy_signal_func_dict.get(strategy_name)
    if signal_func is None:
        raise NotImplementedError(
            f"Signal '{strategy_name}' is not implemented yet."
        )
    
    return signal_func