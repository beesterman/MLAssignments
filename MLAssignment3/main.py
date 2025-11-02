import torch, torchvision,constants, time
from torchvision import transforms
from mlp import mlp
from train import trainModel
from dataHandler import getLoaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device being used is: " + str(device))
torch.manual_seed(constants.seed)
torch.cuda.manual_seed_all(constants.seed)

mlpMNIST = mlp(28*28, 1, 128, 0, 10)
mlpCIFAR10 = mlp(32*32, 1, 128, 0, 10)




mnistTrain, mnistVal, mnistTest = getLoaders("mnist", constants.batchSize[0])


cifarTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
cifar10Data = torchvision.datasets.CIFAR10(root="./MLAssignment3/data", train=True, download=True, transform=cifarTransform)
cifar10Train, cifar10Val = torch.utils.data.random_split(cifar10Data, [.9, .1])

t0 = time.time()
trainModel(mlpMNIST, mnistTrain, mnistVal, constants.learningRates[0], 100, device, constants.optomizer[0])
runtime = (time.time()-t0)/60