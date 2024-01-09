import os
import matplotlib.pyplot as plt

def plot_duration_of_net_exposure(df_g, **params):

    enums = params['enums']

    ranges = {
        "-100 to -80": (-100, -80),
        "-80 to -50": (-80, -50),
        "-50 to -30": (-50, -30),
        "-30 to -10": (-30, -10),
        "-10 to 10": (-10, 10),
        "10 to 30": (10, 30),
        "30 to 50": (30, 50),
        "50 to 80": (50, 80),
        "80 to 100": (80, 100)
    }

    consecutive_lengths = {}
    for k in ranges:
        consecutive_lengths[k] = []

    net_exposures = df_g['net_exposure'].values
    
    last_observed_range = None
    last_observed_range_length = 1
    for val in net_exposures:
        for range_name, (range_start, range_end) in ranges.items():
            
            if (range_start <= val < range_end) or \
                (range_end == 100 and range_start <= val <= range_end):
                
                if last_observed_range == range_name:
                    last_observed_range_length += 1

                else:
                    consecutive_lengths[range_name].append(last_observed_range_length)
                    last_observed_range = range_name
                    last_observed_range_length = 1

    # Assuming your dictionary is called data_dict
    data = list(consecutive_lengths.values())

    fig = plt.figure(figsize=(12, 6))
    plt.boxplot(data)
    plt.xlabel('Net Exposure')
    plt.ylabel('Duration (Days)')
    plt.title('Box Plot of Net Exposure Durations')
    plt.xticks(range(1, len(consecutive_lengths) + 1), consecutive_lengths.keys())

    plt.savefig(os.path.join(enums.STAT_FIGURES_DIR, "NetExposureDurationBoxPlot.png"))
    plt.clf()
    plt.close(fig)