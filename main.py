import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

from sklearn.model_selection import StratifiedKFold

import src
from src.model import FNNModel

import time

BATCH_SIZE = 256 


def k_fold_cross_val(folds: int, dataset):

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for data, label in dataloader:
        X = data
        y = label
    
    # Perform K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert X, y tensors into Tensor dataset, then load it
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

        print(f'\nFold {fold + 1}/{folds}')
        
        # Initialize model, loss, and optimizer
        model = FNNModel()
        opt = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # Train model
        start_time = time.time()
        train_losses = model.fit(train_loader, loss_fn, opt, epochs=5)
        end_time = time.time()

        total_time = end_time - start_time 

        # Validate model
        y_pred = model.predict(X_val)
        val_loss = loss_fn(y_pred, y_val)

        print(f'\n\tval loss {val_loss:.3f}')
        print(f'\ttrain time {total_time:.2f}s')

    

def main():

    # Load KMNIST data
    trainset, testset = src.data_loader.load_data()

    k_fold_cross_val(folds=5, dataset=trainset)

    # Convert into loaders
    #train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    #test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    # Checking the shape of the data
    #data_iter = iter(train_loader)
    #images, labels = next(data_iter)
    #print(f"Image shape: {images.shape}, Labels shape: {labels.shape}")

    #params = {
    #        'optimizer':,
    #        'loss': nn.CrossEntropyLoss(),
    #        'lr': 0.01,
    #        'epochs': 5
    #        }


    #model = FNNModel()
    #loss_fn = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    #epoch_losses, time = model.fit(train_loader, loss_fn, optimizer, epochs=5)

    #print(f'Training time: {time}')

    #outputs, acc = model.predict(test_loader)
    #print(f'test set acc: {acc}')
    

if __name__ == "__main__":
    main()
