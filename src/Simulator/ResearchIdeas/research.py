import os
import logging
import pandas as pd

from src.utils import get_cache_key_for_params
from src.utils import convert_time_columns_to_datetime

from .WinningRatePlotter import WinningRatePlotter
from .conduct_univariate_multivariate_hyp_testing import conduct_univariate_multivariate_hyp_testing

from ..AlphaSimulator.ReportingUtils import get_statistical_summary_of_trades

logger = logging.getLogger("research_logger")

def research(**params):

    '''
    This function performs a research on the potential trades and logs the results.
    It works by loading the potential trades, getting the statistical summary of
    the potential trades and conducting univariate and multivariate
    hypothesis testing.

    Args:
    params: dict
        The parameters for the simulation

    Returns:
        None
    
    '''

    enums = params['enums']
    should_log = params['should_log']


    cache_key = get_cache_key_for_params(params)
    df = pd.read_csv(os.path.join(enums.POTENTIAL_TRADES_DIR, f"{cache_key}.csv"), index_col= 0)
    df = convert_time_columns_to_datetime(df, params['market'])

    df_original = pd.read_csv(os.path.join(enums.POTENTIAL_TRADES_DIR, f"{cache_key}-Original.csv"), index_col= 0)
    df_original = convert_time_columns_to_datetime(df_original, params['market'])

    summaries_df = get_statistical_summary_of_trades(df, df_g=None, **params)
    logger.info(
            "\n----------------------------------------------------------\n" + \
            summaries_df.to_string() + \
            "\n\n**Note:"+ \
            "\n This report is on the all potential trades"
            "\n----------------------------------------------------------\n"
            )

    draw_stat_figures = params["draw_stat_figures"]
    if should_log and draw_stat_figures:
        WinningRatePlotter(df, **params).plot()

    conduct_univariate_multivariate_hyp_testing(df, **params)