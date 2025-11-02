import torch, torchvision, constants
from torchvision import transforms

def printLine(line):
    print(line)
    outFile = open(constants.file, "a")
    outFile.write(line)


def getLoaders(dataset, batchSize):
    if dataset == "mnist":
        trainTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        testTransform = trainTransform
        fullDataSet = torchvision.datasets.MNIST(root="./MLAssignment3/data", train=True, download=True, transform=trainTransform)
        train, val = torch.utils.data.random_split(fullDataSet, [.83, .17])
        test = torchvision.datasets.MNIST(root="./MLAssignment3/data", train=False, download=True, transform=testTransform)

    mk = lambda ds: torch.utils.data.DataLoader(ds, batch_size=batchSize, num_workers=2, pin_memory=True)
    return mk(train), mk(val), mk(test)