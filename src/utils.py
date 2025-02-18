import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
import time

from src.model import FNNModel

def create_optimizer(name, model_params, args):
    '''
    Create an optimizer given dictionary of arguments.
    '''

    if name == 'adam':
        return optim.Adam(model_params, **args)

    elif name == 'adamw':
        return optim.AdamW(model_params, **args)

    elif name == 'rmsprop':
        return optim.RSMprop(model_params, **args)
    else: 
        return None

def k_fold_cross_val(folds, dataset, opt_name, opt_params, train_params):

    # Generate indices for K-folds
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Extract the X, y from data. This needs to be done to split the data using
    # SKLearn's KFold method
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data, label in dataloader:
        X = data
        y = label
    # Scoring metric to keep track of across folds
    accs = []

    # Perform K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        # Set indicies of train and validation data according to K-fold split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert split X, y tensors back into Tensor dataset, then load it
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, train_params['batch_size'], shuffle=True)

        print(f'\n\tFold {fold + 1}/{folds}')

        # Initialize model, loss, and optimizer
        model = FNNModel()
        loss_fn = nn.CrossEntropyLoss()
        opt = create_optimizer(name=opt_name, 
                               model_params=model.parameters(),
                               args=opt_params
                               )

        # Train model
        start_time = time.time()
        model.fit(train_loader, loss_fn, opt, epochs=train_params['epochs'])
        end_time = time.time()

        total_time = end_time - start_time 

        # Validate model
        outputs = model.predict(X_val)
        val_loss = loss_fn(outputs, y_val)

        # Calculate validation accuracy
        _, y_pred = torch.max(outputs, 1)
        correct = (y_pred == y_val).sum().item()
        acc = correct / y_val.size(0) * 100

        print(f'\n\t\tval loss {val_loss:.3f}')
        print(f'\t\tval acc {acc:.2f}%')
        print(f'\t\ttrain time {total_time:.2f}s')

        accs.append(acc)

    # Return the mean accuracy
    return sum(accs)/len(accs)
