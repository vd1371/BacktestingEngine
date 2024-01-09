import os
import pandas as pd
import warnings

def get_all_symbols(**params):

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