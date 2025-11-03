import torch, torchvision, constants
from torch import nn


class mlp(nn.Module): 
    def __init__(self, inputDimension, hiddenLayerNum, hiddenLayerSize, dropRate, numberOfClasses):
        super().__init__()
        layers = []

        #this section of the code creates the sequential layers that are needed to run the nn
        previousLayer = inputDimension
        for i in range(1,hiddenLayerNum):
            layers.append(nn.Linear(previousLayer, hiddenLayerSize))
            layers.append(nn.ReLU(inplace=True))
            if dropRate > 0:
                layers.append(nn.Dropout(dropRate))
            previousLayer = hiddenLayerSize
        layers.append(nn.Linear(previousLayer, numberOfClasses))
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)