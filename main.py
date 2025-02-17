from torch.utils.data import DataLoader


import src


BATCH_SIZE = 32

def main():

    # Load KMNIST data
    trainset, testset = src.data_loader.load_data()

    # Convert into loaders
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    # Checking the shape of the data
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Image shape: {images.shape}, Labels shape: {labels.shape}")




if __name__ == "__main__":
    main()
