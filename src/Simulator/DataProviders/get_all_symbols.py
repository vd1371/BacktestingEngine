import os
import pandas as pd
import warnings

def get_all_symbols(**params):

    '''
    Get all the symbols of the market

    This function returns all the symbols of the market based on the given parameters.
    It reads the symbols from the Symbols.csv file and returns the symbols of the market.
    If the number of symbols is specified, it returns the first n symbols.
    If you want to add more symbols, you can modify the Symbols.csv file.

    Args:
    params: dict
        The parameters for the symbols

    Returns:
    symbols: list
        The symbols of the market
    
    '''

    market = params['market']
    n_symbols = params.get('n_symbols', 200)

    
    symbols_df = pd.read_csv(
        os.path.join("Database", "Symbols.csv")
    )

    symbols = {}

    symbols_of_market = symbols_df.loc[:, market]
    symbols_of_market = symbols_of_market[symbols_of_market.notnull()]

    l = min(n_symbols, len(symbols_of_market))  # noqa: E741
    symbols[market] = symbols_of_market.tolist()[:l]

    return symbols[market]