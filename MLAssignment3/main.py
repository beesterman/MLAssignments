import torch, torchvision,constants, time
from torchvision import transforms
from mlp import mlp
from train import trainModel
from dataHandler import getLoaders, printLine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device being used is: " + str(device))
torch.manual_seed(constants.seed)
torch.cuda.manual_seed_all(constants.seed)


lr = constants.learningRates[1]
batchSize = constants.batchSize[1]
optimizer = constants.optomizer[1]
dropRate = constants.dropRates[0]


mlpMNIST = mlp(28*28, 5, 128, dropRate, 10)
# mlpCIFAR10 = mlp(32*32, 1, 128, droprate, 10)




mnistTrain, mnistVal, mnistTest = getLoaders("mnist", batchSize)


# cifarTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
# cifar10Data = torchvision.datasets.CIFAR10(root="./MLAssignment3/data", train=True, download=True, transform=cifarTransform)
# cifar10Train, cifar10Val = torch.utils.data.random_split(cifar10Data, [.9, .1])

t0 = time.time()
trainedModel, best = trainModel(mlpMNIST, mnistTrain, mnistVal, lr, 20, device, optimizer, 10)
runtime = (time.time()-t0)/60
print(best[0], best[2], runtime)

# printLine("type,LearningRate,BatchSize,Optomizer,DropoutRate,ValidAccuracy,Runtime")
printLine("Deep," + str(lr) + "," + str(batchSize) + "," + str(optimizer) + "," + str(dropRate) + "," + str(best[0]) + "," + str(runtime))