import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import LONG, SHORT

class WinningRatePlotter:

    '''
    ## GUIDE: Step 11

    This class is responsible for plotting the winning rate of the trades.

    The plots are saved in the STAT_FIGURES_DIR directory.
    '''

    def __init__(self, report_dfs, **params) -> None:

        plt.rcdefaults()
        
        self.market = params.get("market")
        self.strategy_name = params['strategy_name']
        self.strategy_type = params['strategy_type']
        self.enums = params['enums']

        self.df = report_dfs

    def plot(self):
        self._draw_winning_rate_for_weekdays()
        self._draw_stat_ratios()
        self._draw_positive_stats()
        self._draw_histogram_of_pnl_ratios()
        self._draw_winning_rate_for_opening_hours()


    def _draw_winning_rate_for_weekdays(self):

        col = "stat_weekday"
        if col not in self.df.columns:
            return

        grouped = self.df.groupby([col, 'trade_direction'], observed=False)['is_successful'].mean().unstack()

        # create a figure with two axes
        fig, ax = plt.subplots()
        for y in np.arange(0.1, 1, 0.1):
            ax.axhline(y=y, color=(0.8, 0.8, 0.8), linestyle='--', linewidth=0.5)

        width = 0.25  # the width of the bars
        multiplier = 0

        # set the width of the bars
        width = 0.35

        # create the first set of bars (LONG trades)
        x1 = np.arange(len(grouped.index))
        ax.bar(x1 - width/2, grouped[LONG], width=width, color='blue', label='LONG')

        # create the second set of bars (SHORT trades)
        if self.strategy_type == "market_neutral" and SHORT in grouped.columns:
            x2 = np.arange(len(grouped.index))
            ax.bar(x2 + width/2, grouped[SHORT], width=width, color='red', label='SHORT')

        # add labels and title
        ax.set_xlabel(col)
        ax.set_ylabel('Probability of Success')
        ax.set_title(f'Probability of Success by {col} and Trade Direction')

        # add x-axis tick labels
        ax.set_xticks(np.arange(len(grouped.index)))
        ax.set_xticklabels(grouped.index)
        ax.set_ylim(0, 1)

        # add a legend
        ax.legend()

        # display the chart
        plt.savefig(os.path.join(self.enums.STAT_FIGURES_DIR, f"{col}.png"))
        plt.clf()
        plt.close(fig)

    def _draw_winning_rate_for_opening_hours(self):
        
        col = "stat_opening_hour"
        if col not in self.df.columns:
            return
        grouped = self.df.groupby([col, 'trade_direction'], observed=False)['is_successful'].agg(['mean', 'count']).unstack()

        # create a figure with two axes
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))
        for y in np.arange(0.1, 1, 0.1):
            ax1.axhline(y=y, color=(0.8, 0.8, 0.8), linestyle='--', linewidth=0.5)

        width = 0.25  # the width of the bars
        multiplier = 0

        # create the first set of bars (LONG trades)
        x1 = np.arange(len(grouped.index))
        x2 = np.arange(len(grouped.index))

        ax1.bar(x1 - width/2, grouped[('mean', LONG)], width=width, color='blue', label='LONG')
        ax2.bar(x2 - width/2, grouped[('count', LONG)], width=width, color='lightblue', label='LONG')

        # create the second set of bars (SHORT trades)
        if self.strategy_type == "market_neutral" and ('mean', SHORT) in grouped.columns: 
            ax1.bar(x1 + width/2, grouped[('mean', SHORT)], width=width, color='red', label='SHORT')
            ax2.bar(x2 + width/2, grouped[('count', SHORT)], width=width, color='salmon', label='SHORT')

        # add labels and title
        ax1.set_xlabel(col)
        ax1.set_ylabel('Probability of Success')
        ax1.set_title(f'Probability of Success by {col} and Trade Direction')

        # add x-axis tick labels
        ax1.set_xticks(np.arange(len(grouped.index)))
        ax1.set_xticklabels(grouped.index)
        ax1.set_ylim(0, 1)

        # add a legend
        ax1.legend()

        # add labels and title
        ax2.set_xlabel(col)
        ax2.set_ylabel('N occurrences')
        ax2.set_title(f'N occurrences by {col} and Trade Direction')

        # add x-axis tick labels
        ax2.set_xticks(np.arange(len(grouped.index)))
        ax2.set_xticklabels(grouped.index)

        # add a legend
        ax2.legend()

        # display the chart
        plt.savefig(os.path.join(self.enums.STAT_FIGURES_DIR, f"{col}.png"))
        
        plt.clf()
        plt.close(fig)

    def _draw_stat_ratios(self):
        """
        Draw bar charts based on statistical ratios in the DataFrame.

        This method iterates over each column in the DataFrame and checks if
            the column name contains both "stat" and "ratio".
        If a column matches this condition, it creates a new column called
            'ratio_group' that categorizes the values based on predefined
            bins and labels.
        It then calls the _draw_bar_chart method to draw a bar chart
            for the column.

        Note:
        - This method assumes that the DataFrame is stored in the attribute
            'df' of the class instance.

        Returns:
        None
        """
        cols_to_plot = []
        targets = ['ratio', 'Delta', 'corr_', 'CMF', 'AutoCorr', "beta"]
        for col in self.df.columns:
            for target in targets:
                if "stat" in col and target in col:
                    cols_to_plot.append(col)
            
        for col in cols_to_plot:
            try:
                bins, labels = generate_bins_labels(self.df[col], num_bins=13)

                # create a new column with the bin labels
                self.df['ratio_group'] = pd.cut(
                    self.df[col],
                    bins=bins,
                    labels=labels
                    )
                
                _draw_bar_chart(self, col)

            except Exception as e:
                print (col, "has a problem.")
                print (e)
                pass

    def _draw_positive_stats(self):
        cols_to_plot = []
        targets = ['R2_', '_Vola', 'ATR', 'RSI', 'ADX', 'DMP', 'DMN', 'MFI', 'volatility_forecast', "_IV"]
        for col in self.df.columns:
            for target in targets:
                if "stat" in col and target in col and "signal" not in col and "ratio" not in col:
                    cols_to_plot.append(col)
        
        for col in cols_to_plot:

            bins, labels = generate_bins_labels_for_positive_values(self.df[col], num_bins=13)

            # create a new column with the bin labels
            self.df['ratio_group'] = pd.cut(
                self.df[col],
                bins=bins,
                labels=labels
                )
            
            _draw_bar_chart(self, col)


    def _draw_histogram_of_pnl_ratios(self):
        # Filter the DataFrame based on trade_direction
        positive_df = self.df[self.df['trade_direction'] == 1]
        negative_df = self.df[self.df['trade_direction'] == -1]

        # Create a figure and axes to hold the histograms
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))

        bin_range = (-0.1, 0.1)
        # Plot the histograms
        ax1.hist(positive_df['PnL_ratio'], bins=20, alpha=0.5,
                range=bin_range, label='Trade Direction: 1')
        ax2.hist(negative_df['PnL_ratio'], bins=20, alpha=0.5,
                 range=bin_range, label='Trade Direction: -1')

        # Set labels and title
        for ax in [ax1, ax2]:
            ax.set_xlabel('PnL Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title('Histograms of PnL Ratio by Trade Direction')

            # Add a legend
            ax.legend()

        plt.savefig(os.path.join(self.enums.STAT_FIGURES_DIR, "PnL_ratio.png"))
        plt.clf()
        plt.close(fig)


def _draw_bar_chart(self, col):
    '''
    HINT: Make sure the number of bins are even numbers.
    
    '''
    file_name = os.path.join(self.enums.STAT_FIGURES_DIR, f"{col}.png")
    if os.path.exists(file_name):
        print (f"{col} figure already exists.")
        return

    # create a figure with one axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    
    # ------------------------------------------------------------------------ #
    #                             is_successful                                #
    # ------------------------------------------------------------------------ #
    # group by weekday, trade_direction, and ratio_group, and calculate the mean of is_successful
    grouped = self.df.groupby(['trade_direction', 'ratio_group'], observed=False)['is_successful'].mean().unstack()

    positive_colors = list(plt.cm.Blues(np.linspace(0.2, 0.8, int(len(grouped.columns)/2))))
    negative_colors = list(plt.cm.Reds(np.linspace(0.2, 0.8, int(len(grouped.columns)/2))))
    colors = negative_colors +  list(reversed(positive_colors))

    for y in np.arange(0.1, 1, 0.1):
        ax1.axhline(y=y, color=(0.8, 0.8, 0.8), linestyle='--', linewidth=0.5)
    for y in [0.45, 0.55]:
        ax1.axhline(y=y, color=(0.6, 0.6, 0.6), linestyle='--', linewidth=0.5)
    ax1.axhline(y=0.5, color=(0.4, 0.4, 0.4), linestyle='--', linewidth=0.5)

    # set the width of the bars
    width = 0.35

    # create the bars for each ratio group
    labels = grouped.index.unique()
    x = np.arange(len(labels))
    for i, group in enumerate(grouped.columns):
        ax1.bar(
            x - width/2 + i*width/float(len(grouped.columns)),
            grouped[group].values,
            width=width/float(len(grouped.columns)),
            label=group,
            color=colors[i]
            )

    # add x-axis tick labels
    ax1.set_title(col)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-0.5, len(labels)-0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    # add labels and title
    ax1.set_ylabel('Probability of Success')
    ax1.legend()

    # ------------------------------------------------------------------------ #
    #                the count of occurrences on the twin axis                 #
    # ------------------------------------------------------------------------ #
    total_counts = self.df.groupby(['trade_direction'], observed=False)['ratio_group'].count()
    count_grouped = self.df.groupby(['trade_direction', 'ratio_group'], observed=False)['ratio_group'].count().unstack()

    trade_directions = [LONG]
    if self.strategy_type == "market_neutral" and SHORT in count_grouped.index:
        trade_directions.append(SHORT)


    for i, td in enumerate(count_grouped.columns):
        ax3.bar(
            x - width/2 + i*width/float(len(count_grouped.columns)),
            count_grouped[td].values,
            width=width/float(len(count_grouped.columns)),
            label=f"{td} (count of occurrences)",
            color=colors[i]
        )

    # add labels and title to the twin axis
    ax3.set_ylabel('N Occurrences')
    ax3.set_xlim(-0.5, len(labels)-0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    # ax3.legend()

    # ------------------------------------------------------------------------ #
    #                             the PnL_ratio mean                           #
    # ------------------------------------------------------------------------ #
    grouped = self.df.groupby(['trade_direction', 'ratio_group'], observed=False)['PnL_ratio'].mean().unstack()

    positive_colors = list(plt.cm.Blues(np.linspace(0.2, 0.8, int(len(grouped.columns)/2))))
    negative_colors = list(plt.cm.Reds(np.linspace(0.2, 0.8, int(len(grouped.columns)/2))))
    colors = negative_colors +  list(reversed(positive_colors))

    # set the width of the bars
    width = 0.35

    # create the bars for each ratio group
    labels = grouped.index.unique()
    x = np.arange(len(labels))
    for i, group in enumerate(grouped.columns):
        ax2.bar(
            x - width/2 + i*width/float(len(grouped.columns)),
            grouped[group].values,
            width=width/float(len(grouped.columns)),
            label=group,
            color=colors[i]
            )

    # add x-axis tick labels
    ax2.set_xlim(-0.5, len(labels)-0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    # add labels and title
    ax2.set_ylabel('EV[PnL_ratio]')

    # ------------------------------------------------------------------------ #
    #                            the sum of PnL_ratio                          #
    # ------------------------------------------------------------------------ #
    grouped = self.df.groupby(['trade_direction', 'ratio_group'], observed=False)['PnL_ratio'].sum().unstack()

    positive_colors = list(plt.cm.Blues(np.linspace(0.2, 0.8, int(len(grouped.columns)/2))))
    negative_colors = list(plt.cm.Reds(np.linspace(0.2, 0.8, int(len(grouped.columns)/2))))
    colors = negative_colors +  list(reversed(positive_colors))

    # set the width of the bars
    width = 0.35

    # create the bars for each ratio group
    labels = grouped.index.unique()
    x = np.arange(len(labels))
    for i, group in enumerate(grouped.columns):
        ax4.bar(
            x - width/2 + i*width/float(len(grouped.columns)),
            grouped[group].values,
            width=width/float(len(grouped.columns)),
            label=group,
            color=colors[i]
            )

    # add x-axis tick labels
    ax4.set_xlim(-0.5, len(labels)-0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    # add labels and title
    ax4.set_ylabel('Sum[PnL_ratio]')



    # display the chart and save
    plt.savefig(file_name)
    plt.clf()
    plt.close("all")



def generate_bins_labels(data, num_bins):
    '''
    num_bins must be odd
    '''
    assert num_bins % 2 == 1

    min_val = data.mean() - 3*data.std()
    max_val = data.mean() + 3*data.std()

    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins)]

    n = num_bins//2
    bins = [0]
    bins = [-(i+1) * bin_width for i in range(n)][::-1] + bins + [(i+1) * bin_width for i in range(n)]


    labels = [f'{bins[i]:.4f} to {bins[i+1]:.4f}' for i in range(num_bins-1)]
    labels.insert(0, f'<= {bins[0]:.4f}')
    labels.append(f'>= {bins[-1]:.4f}')

    bins = [-np.inf] + bins + [np.inf]
    
    return bins, labels

def generate_bins_labels_for_positive_values(data, num_bins):
    '''
    num_bins must be odd
    '''
    assert num_bins % 2 == 1

    min_val = data.median() - 3*data.std()
    max_val = data.median() + 3*data.std()

    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins)]

    bins = [i * bin_width for i in range(num_bins)]

    labels = [f'{bins[i]:.4f} to {bins[i+1]:.4f}' for i in range(num_bins-1)]
    labels.insert(0, f'<= {bins[0]:.4f}')
    labels.append(f'>= {bins[-1]:.4f}')

    bins = [-np.inf] + bins + [np.inf]
    
    return bins, labels