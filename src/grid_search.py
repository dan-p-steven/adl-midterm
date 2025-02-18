import itertools
import numpy as np
import json

from src.utils import k_fold_cross_val

def grid_search(dataset, args):
    '''
    Perform grid search on dataset to find optimal hyperparameters. Generates
    a model with the ADL midterm architecture for an optimizer, and uses hyper- 
    parameters ranges defined in the args variable.

    Saves the best and current hyperparameters to files under ./models/. This
    is done for checkpoint purposes during a long search.

    Returns the best accuracy, optimizer hyperparameters, and training hyperparameters.
    '''

    # Generate all combinations of hyperparameters from a hyperparameter grid.
    opt_combs = itertools.product(*args['optimizer_param_grid'].values())
    opt_fields = list(args['optimizer_param_grid'].keys())

    train_combs = itertools.cycle(itertools.product(*args['training_param_grid'].values()))
    train_fields = list(args['training_param_grid'].keys())

    # Calculate the size of the hyperparameter search space.
    size = 1
    training_params_size = 1

    for ranges in args['optimizer_param_grid'].values():
        size *= len(ranges)

    for ranges in args['training_param_grid'].values():
        size *= len(ranges)
        training_params_size *= len(ranges)


    # Variables to keep track of the best model.
    best_score = 0
    best_opt_params = {}
    best_train_params = {}

    # Counter to keep track of hyperparameter set number.
    i = 0

    # Iterate over all combinations of optimizer parameters.
    for comb in opt_combs:

        # Create the optimizer param dictionary.
        opt_params = dict(zip(opt_fields, comb))

        # Iterate over all combinations of training parameters.
        for _ in range(training_params_size):

            # Create the training param dictionary.
            train_params = dict(zip(train_fields, next(train_combs)))

            # Increment hyperparameter space counter.
            i += 1

            print (f'\nHyperparameter set [{i}/{size}]')
            print(opt_params, train_params)

            # Perform KFCV
            score = k_fold_cross_val(5, dataset, args['optimizer'], opt_params, train_params)

            # Update the best scores, optimizer parameters and training parameters
            # if score is the best.
            if score > best_score:
                best_score = score
                best_opt_params = opt_params
                best_train_params = train_params

                best_data = [best_score, best_opt_params, best_train_params]

                # Write best hyperparam data to a file, for checkpoint purposes.
                with open('./models/best.json', 'w') as file:
                    json.dump(best_data, file)

            # Write the current hyperparam data to a file for checkpoint purposes.
            current_data = [score, opt_params, train_params]
            with open('./models/current.json', 'w') as file:
                json.dump(current_data, file)

    return best_score, best_opt_params, best_train_params
