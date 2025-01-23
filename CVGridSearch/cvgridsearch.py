import itertools


def cv_grid_search(model, parameter_search_space, folds):
    # Generate all combinations of parameters
    keys, values = zip(*parameter_search_space.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    # Print all combinations
    for combo in combinations:
        print(combo)
    # For each combination of model parameters do n fold cv

    # Best model is that which has 