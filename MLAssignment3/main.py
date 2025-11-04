import torch, torchvision,constants, time
from torchvision import transforms
from mlp import mlp
from train import trainModel, evaluate
from dataHandler import getLoaders, printLine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device being used is: " + str(device))
torch.manual_seed(constants.seed)
torch.cuda.manual_seed_all(constants.seed)


lr = constants.learningRates[2]
batchSize = constants.batchSize[2]
optimizer = constants.optomizer[1]
dropRate = constants.dropRates[2]

#TODO run cifar models decending
# mlpMNIST = mlp(28*28, 5, 128, dropRate, 10)
mlpCIFAR10 = mlp(3*32*32, 5, 128, dropRate, 10)




train, val, test = getLoaders("anythingelse", batchSize)


# t0 = time.time()
trainedModel, best = trainModel(mlpCIFAR10, train, val, lr, 20, device, optimizer, 10)
# runtime = (time.time()-t0)/60
# print(best[0], best[2], runtime)

# printLine("type,LearningRate,BatchSize,Optomizer,DropoutRate,ValidAccuracy,Runtime")
# printLine("Deep," + str(lr) + "," + str(batchSize) + "," + str(optimizer) + "," + str(dropRate) + "," + str(best[0]) + "," + str(runtime))

finalAccuracy = evaluate(trainedModel,test,device)
print(finalAccuracy)