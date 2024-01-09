import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

def plot_the_budget_vs_time(df_g, market_index, TRADE_REPORTS_DIR, **params):

    strategy_name = params["strategy_name"]
    market = params['market']

    trading_interval = params.get("trading_interval")

    df_g_tmp = df_g.copy()
    df_g_tmp.index = df_g_tmp.index.date

    if market_index is not None:
        market_index_tmp = market_index.copy()
        market_index_tmp.index = market_index_tmp.index.date

        if trading_interval == "1d":
            df_tmp = pd.merge(df_g_tmp, market_index_tmp, left_index=True, right_index=True)
        else:
            df_tmp = pd.merge(df_g, market_index, left_index=True, right_index=True)

    else:
        df_tmp = df_g_tmp

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(df_tmp['PortfolioValue'], 'g-', label='Balance', linewidth=0.5)
    ax1.set_title(f"{strategy_name}")
    fig.set_size_inches(11.69, 8.27/3)
    fig.tight_layout()

    if market_index is not None:
        ax2.plot(df_tmp['Close'], 'b-', label=f"{market}-Index", linewidth=0.5)

    ax1.set_xlabel('Dates')
    ax1.set_ylabel('Balance', color='g')

    if market_index is not None:
        ax2.set_ylabel(f"{market}-Index", color='b')
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)

    plt.savefig(os.path.join(TRADE_REPORTS_DIR, "BudgetPlot.png"))
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image_data = buffer.read()
    base64_encoded = base64.b64encode(image_data).decode('utf-8')

    plt.clf()

    return base64_encoded