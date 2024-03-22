import numpy as np
import itertools
import time
from copy import deepcopy
import pandas as pd
import os

from .simulate import simulate


def optimize(**params):

    enums = params['enums']

    trading_params_ranges = {
        "stop_loss_percentage": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "risk_level_percentage": [0.1, 0.2, 0.3, 0.4, 0.5],
        "should_stop_loss": [True, False],

        "take_profit_percentage": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "should_take_profit": [True, False],
    }

    opt_method = params.get("opt_method", "grid_search")

    params_to_pass = deepcopy(params)
    params_to_pass.update({
        "should_log": False,
        "should_load_from_cache": False,
        })

    if opt_method == "grid_search":
        results = _optimize_using_grid_search(trading_params_ranges, params_to_pass)

    elif opt_method == "random_search":
        n_random_search = params.get("n_random_sampling", 10)
        results = _optimize_using_random_search(trading_params_ranges, n_random_search, params_to_pass)
    
    elif opt_method == "genetic_algorithm":
        results = _optimize_using_genetic_algorithm(trading_params_ranges, params_to_pass)

    else:
        raise ValueError(f"Unknown optimization method: {opt_method}")

    # Convert the results into dataframe
    holder = []
    for combination, summary in results:
        holder.append({**combination, **summary})
    
    results_df = pd.DataFrame(holder)

    col = "sharpe" if "sharpe" in results_df.columns else "annual(%)"
    results_df = results_df.sort_values(by=col, ascending=False)

    direc = os.path.join(enums.OPTIMIZATION_RESULTS_DIR, f"{opt_method}.csv")
    results_df.to_csv(direc)

    return




def _optimize_using_grid_search(trading_params_ranges, params):
    
    print (f"GRID SEARCH: Optimizing the following parameters: {trading_params_ranges.keys()}")

    # Create the grid
    grid = []
    for key in trading_params_ranges:
        grid.append(trading_params_ranges[key])

    # Create the combinations
    combinations = []
    for combination in itertools.product(*grid):
        combinations.append(dict(zip(trading_params_ranges, combination)))

    opt_result_holder = []

    start = time.time()
    for i, combination in enumerate(combinations):
    
        params_to_pass = deepcopy(params)
        params_to_pass.update(combination)

        reports_df, summary = simulate(**params_to_pass)
        opt_result_holder.append((combination, summary.iloc[0, :].to_dict()))

        print (f"Grid search iteration {i+1}/{len(combinations)} in {time.time() - start:.2f} seconds")
        start = time.time()

    return opt_result_holder


def _optimize_using_random_search(trading_params_ranges, n_random_search, params):
    
    print (f"RANDOM SEARCH: Optimizing the following parameters: {trading_params_ranges.keys()}")

    # Create the grid
    grid = []
    for key in trading_params_ranges:
        grid.append(trading_params_ranges[key])

    # Create the combinations
    combinations = []
    for combination in itertools.product(*grid):
        combinations.append(dict(zip(trading_params_ranges, combination)))

    opt_result_holder = []

    if n_random_search > len(combinations):
        raise ValueError(f"n_random_search ({n_random_search}) is greater than the number of combinations ({len(combinations)})")

    combinations = np.random.choice(combinations, n_random_search, replace=False)

    start = time.time()
    for i, combination in enumerate(combinations):

        params_to_pass = deepcopy(params)
        params_to_pass.update(combination)

        reports_df, summary = simulate(**params_to_pass)
        opt_result_holder.append((combination, summary.iloc[0, :].to_dict()))

        print (f"Random search iteration {i+1}/{len(combinations)} in {time.time() - start:.2f} seconds")
        start = time.time()

    return opt_result_holder


def _optimize_using_genetic_algorithm(trading_params_ranges, params):

    '''
    IMPORTANT:
        This function is not debugged yet. There could be some issues with the implementation.
    '''

    print (f"GENETIC ALGORITHM: Optimizing the following parameters: {trading_params_ranges.keys()}")

    # Create the grid
    grid = []
    for key in trading_params_ranges:
        grid.append(trading_params_ranges[key])
    
    # Genetic Algorithm Parameters
    population_size = 5
    num_generations = 2
    crossover_rate = 0.8
    mutation_rate = 0.1
    n_elites = 2

    # Create the combinations
    combinations = []
    for combination in itertools.product(*grid):
        combinations.append(dict(zip(trading_params_ranges, combination)))

    if population_size * num_generations > len(combinations):
        raise ValueError(f"population x num_generations is greater than the number of combinations ({len(combinations)})")

    # Randomly initialize the initial population
    population = []
    for _ in range(population_size):
        individual = np.random.choice(combinations)
        population.append(individual)

    start = time.time()

    for generation in range(num_generations):
        print(f"Generation {generation+1}/{num_generations}")
        offspring = []

        # Evaluate the fitness of the population
        fitness_scores = []
        population_results = []
        for i, individual in enumerate(population):
            print (f"Generation {generation+1}: Evaluating individual {i+1}/{len(population)}")

            params_to_pass = deepcopy(params)
            params_to_pass.update(individual)
            reports_df, summary = simulate(**params_to_pass)
        
            summary_dict = summary.iloc[0, :].to_dict()
            population_results.append((individual, summary_dict))
            if summary_dict.get("sharpe") is not None:
                score = summary_dict.get("sharpe")

            else:
                score = summary_dict.get("annual(%)")

            fitness_scores.append(score)

        best_individual = population_results[np.argmax(fitness_scores)]
        print(f"Best individual in generation {generation+1}: {best_individual}")

        # Select the best individuals for the next generation
        offspring = elitism_selection(population, fitness_scores, n_elites)

        # Select parents for reproduction
        parents = tournament_selection(population, fitness_scores, n_elites)

        # Generate offspring through crossover and mutation
        for i in range(len(parents)):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]  # Wrap around for the last parent

            if np.random.random() < crossover_rate:
                offspring.append(crossover(parent1, parent2))
            else:
                offspring.append(parent1)

        # Apply mutation to the offspring
        for i in range(len(offspring)):
            if i < n_elites:
                continue
            if np.random.random() < mutation_rate:
                offspring[i] = mutation(offspring[i], trading_params_ranges)

        population = offspring[:population_size]
        

    print(f"Total search time: {time.time() - start:.2f} seconds")
    return population_results


def tournament_selection(population, fitness_scores, tournament_size=5, n_elites=2):
    parents = []
    for _ in range(len(population)):
        tournament = np.random.choice(range(len(population)), tournament_size, replace=True)
        best_individual = max(tournament, key=lambda x: fitness_scores[x])
        parents.append(population[best_individual])
    return parents[n_elites:]


def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        if np.random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child


def mutation(individual, grid):
    
    mutated_individual = individual.copy()
    key_to_mutate = np.random.choice(list(mutated_individual.keys()))
    mutated_individual[key_to_mutate] = np.random.choice(grid[key_to_mutate])

    return mutated_individual


def elitism_selection(population, fitness_scores, n_elites):
    
    # Find the argmax of the fitness scores n_elites times
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:n_elites]