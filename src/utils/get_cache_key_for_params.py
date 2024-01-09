
def get_cache_key_for_params(params):

    market = params['market']
    unique_keys = [
        'n_symbols',
        'risk_level_percentage',
        'stop_loss_percentage',
        'trading_interval',
        'years_to_consider',
        'take_profit_percentage',
        'should_close_at_end_of_candle',
        'should_stop_loss',
        'should_take_profit',
        'should_limit_one_position_in_run_alpha',
    ]
    
    cache_key = market
    for k in unique_keys:
        val = params[k]
        if isinstance(val, bool): val = str(int(val))
        else: val = str(val)

        if cache_key == "":
            cache_key = val
        else:
            cache_key += "-" + val

    return cache_key