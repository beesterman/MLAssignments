import torch, torchvision, constants

def trainModel(model, trainSet, valSet, learningRate, epochs, device, optimizer):
    model.to(device)
    if optimizer == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=learningRate)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=learningRate)
    criteria = torch.nn.CrossEntropyLoss()
    