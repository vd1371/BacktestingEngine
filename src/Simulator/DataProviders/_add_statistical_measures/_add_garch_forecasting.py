from arch import arch_model


def _add_garch_forecasting(df, interval, **params):
    """
    Add GARCH forecasting results to the dataframe.
    """

    ##TODO: ASSIGNMENT #4 - Add GARCH forecasting here

    sholud_add_garch = params['should_add_garch']

    if not sholud_add_garch:
        return df

    df = df.copy()
    
    ## TODO: Add GARCH forecasting

    return df