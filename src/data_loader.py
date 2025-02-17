from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import KMNIST

def load_data():


    # Construct toTensor transform.
    transform = transforms.Compose([transforms.ToTensor()])

    # Load trainset into giant batch.
    trainset = KMNIST(root='../data', train=True, download=True, transform=transform)

    # Load the training set into one giant batch. This is done to calculate
    # mean and std for the entire training set.
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)


    data = next(iter(trainloader))

    # Calculate mean and std
    mean = data[0].mean()
    stddev = data[0].std()

    # Create a Normalization transform.
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, stddev)])

    # Re-download the datasets with Normalization applied.
    trainset = KMNIST(root='../data', train=True, download=True, transform=transform)
    testset = KMNIST(root='../data', train=False, download=True, transform=transform)

    return trainset, testset
