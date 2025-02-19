import torch
import json
import torch.nn as nn

import src
from src.model import FNNModel
from src.data_loader import load_data
from src.grid_search import grid_search
from src.utils import create_optimizer, final_train_and_evaluate, k_fold_cross_val


ARGS = {
        #'optimizer': 'adam',
        'optimizer': 'adamw',
        #'optimizer': 'rmsprop',

        'optimizer_param_grid': {
            'lr': [0.0001, 0.001, 0.01],
            'betas': [(0.9, 0.999), (0.5, 0.999)], # remove for rmsprop :)
            'weight_decay': [1e-4, 1e-2]
            },

        'training_param_grid': {
            'epochs': [5, 10, 20],
            'batch_size': [128]
            }
}


def main():

    # Load KMNIST data
    trainset, testset = load_data()

    # Perform grid search
    score, opt_params, train_params = grid_search(dataset=trainset, args=ARGS)
    print (f'\n\nBEST OVERALL VAL ACCURACY: {score}%\nOPTIMIZER PARAMS: {opt_params}\nTRAINING PARAMS: {train_params}\n')

    print(f'\nTRAINING AND EVALUATING FINAL MODEL W/ BEST PARAMS ...')
    # 1. Create and train model with optimal params
    model, epoch_losses, epoch_accs, train_time = final_train_and_evaluate(trainset, 
                                                                           testset, 
                                                                           opt_name=ARGS['optimizer'], 
                                                                           opt_params=opt_params,
                                                                           train_params=train_params)

    # Write model weights to file
    print(f'\nModel weights saved to '+f'./models/{ARGS["optimizer"]}.pth')
    torch.save(model.state_dict(), f'./models/{ARGS["optimizer"]}.pth')

    # Write model metrics to file as well.
    print(f'Model metrics saved to '+f'./models/{ARGS["optimizer"]}_metrics.json')
    with open(f'./models/{ARGS["optimizer"]}_metrics.json', "w") as f:
        json.dump([epoch_losses, epoch_accs, train_time], f, indent=4)

if __name__ == "__main__":
    main()
