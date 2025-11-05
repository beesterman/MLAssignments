import torch, torchvision, constants
from torchvision import transforms

def printLine(line):
    print(line)
    outFile = open(constants.file, "a")
    outFile.write(line + "\n")


def getLoaders(dataset, batchSize):
    if dataset == "mnist":
        mnistTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        fullDataSet = torchvision.datasets.MNIST(root="./MLAssignment3/data", train=True, download=True, transform=mnistTransform)
        train, val = torch.utils.data.random_split(fullDataSet, [.83, .17])
        test = torchvision.datasets.MNIST(root="./MLAssignment3/data", train=False, download=True, transform=mnistTransform)
    if dataset == "mnistTest":
        mnistTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        fullDataSet = torchvision.datasets.MNIST(root="./MLAssignment3/data", train=True, download=True, transform=mnistTransform)
        train, val = torch.utils.data.random_split(fullDataSet, [.83, .17])
        test = torchvision.datasets.MNIST(root="./MLAssignment3/data", train=False, download=True, transform=mnistTransform)
        train = fullDataSet
    if dataset == "cifar10Test":
        cifarTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
        fullDataSet = torchvision.datasets.CIFAR10(root="./MLAssignment3/data", train=True, download=True, transform=cifarTransform)
        train, val = torch.utils.data.random_split(fullDataSet, [.9, .1])
        test = torchvision.datasets.CIFAR10(root="./MLAssignment3/data", train=False, download=True, transform=cifarTransform)
        train = fullDataSet
    else:
        cifarTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
        fullDataSet = torchvision.datasets.CIFAR10(root="./MLAssignment3/data", train=True, download=True, transform=cifarTransform)
        train, val = torch.utils.data.random_split(fullDataSet, [.9, .1])
        test = torchvision.datasets.CIFAR10(root="./MLAssignment3/data", train=False, download=True, transform=cifarTransform)

    mk = lambda ds: torch.utils.data.DataLoader(ds, batch_size=batchSize, pin_memory=True)
    return mk(train), mk(val), mk(test)