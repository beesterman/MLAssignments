import torch, torchvision, constants

def trainModel(model, trainSet, valSet, learningRate, epochs, device, optimizer, stoppingFactor):
    print("Training Begining")
    model.to(device)
    if optimizer == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=learningRate)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=learningRate)
    criteria = torch.nn.CrossEntropyLoss()
    #best index 0 = accuracy, 1 = model state, 2 = epoch
    best = [0, None, 0]
    noImprove = 0
    for epoc in range(1, epochs+1):
        for x,y in trainSet:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criteria(logits, y)
            loss.backward()
            opt.step()
        
        valAccuracy = evaluate(model, valSet, device)
        if valAccuracy > best[0]:
            best = [valAccuracy, model.state_dict(), epoc]
            noImprove = 0
        else:
            noImprove += 1
        if noImprove == stoppingFactor:
            break
        print(epoc, best[0], valAccuracy)

    if best[1] != None:
        model.load_state_dict(best[1])
    return model, best


def evaluate(model, valSet, device):
    model.eval()
    correct = 0
    n = 0
    for x,y in valSet:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
    return correct / n