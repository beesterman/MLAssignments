import torch, torchvision
from torchvision.datasets import MNIST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device being used is: " + str(device))

