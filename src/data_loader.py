from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import KMNIST

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = KMNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    data = next(iter(trainloader))
    mean = data[0].mean()
    stddev = data[0].std()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, stddev)])

    trainset = KMNIST(root='../data', train=True, download=True, transform=transform)
    testset = KMNIST(root='../data', train=False, download=True, transform=transform)

    return trainset, testset