import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


import src
from src.model import FNNModel


BATCH_SIZE = 256 

def main():

    # Load KMNIST data
    trainset, testset = src.data_loader.load_data()

    # Convert into loaders
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    #test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    # Checking the shape of the data
    #data_iter = iter(train_loader)
    #images, labels = next(data_iter)
    #print(f"Image shape: {images.shape}, Labels shape: {labels.shape}")


    model = FNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.fit(train_loader, criterion, optimizer)

    

if __name__ == "__main__":
    main()
