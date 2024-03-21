import os
import sys
import cProfile
from params import get_params
from config import ENUMS
from src import simulate, research, analyze_raw_data, optimize
from src.utils import submit_loggers

def run(run_mode="simulate"):

    '''
    Run the simulation, research or analyze raw data

    Args:
    run_mode: str
        The mode to run the simulation, research or analyze raw data
    
    '''

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    params = get_params()
    enums = ENUMS(**params)
    submit_loggers(enums)

    if run_mode == "simulate":
        simulate(enums=enums, **params)

    elif run_mode == "research":
        research(enums=enums, **params)

    elif run_mode == "analyze_raw_data":
        analyze_raw_data("1d", enums=enums, **params)

    elif run_mode == "optimize":
        optimize(opt_method="genetic_algorithm" , enums=enums, **params)

if __name__ == "__main__":
    profile = cProfile.Profile()
    profile.enable()

    ## GUIDE: Step 1

    # run(run_mode="analyze_raw_data")
    run(run_mode="simulate")
    # run(run_mode="research")
    # run(run_mode="optimize")

    profile.disable()
    profile.dump_stats(os.path.join("reports", "profile.prof"))