import src
from src.model import FNNModel
from src.data_loader import load_data
from src.grid_search import grid_search
from src.utils import create_optimizer, k_fold_cross_val


ARGS = {
        'optimizer': 'adam',
        #'optimizer': 'adamw',
        #'optimizer': 'rmsprop',

        'optimizer_param_grid': {
            'lr': [1],
            'eps': [1e-8]
            },

        'training_param_grid': {
            'epochs': [5],
            'batch_size': [128]
            }
}



def main():

    # Load KMNIST data
    trainset, testset = load_data()

    # Perform grid search
    score, opt_params, train_params = grid_search(dataset=trainset, args=ARGS)

    print (f'\n\nBEST OVERALL ACCURACY: {score}%\nOPTIMIZER PARAMS: {opt_params}\nTRAINING PARAMS: {train_params}\n')


if __name__ == "__main__":
    main()
