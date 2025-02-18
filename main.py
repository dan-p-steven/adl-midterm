import torch.nn as nn

import src
from src.model import FNNModel
from src.data_loader import load_data
from src.grid_search import grid_search
from src.utils import create_optimizer, k_fold_cross_val, final_train, final_evaluate


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
    print (f'\n\nBEST OVERALL ACCURACY: {score}%\nOPTIMIZER PARAMS: {opt_params}\nTRAINING PARAMS: {train_params}\n')

    # TODO:
    # 1. Create and train model with optimal params
    #model, epoch_losses, train_time = final_train(...)

    # 2. Evaluate the model on test set
    #acc, _ = final_evaluate(model, nn.CrossEntropyLoss(), X_test, y_test)

    # 3. Compare and contrast across optimizers


if __name__ == "__main__":
    main()
