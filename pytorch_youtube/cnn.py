# Imports
import torch
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
# All functions that don't have any parameters (relu, ...)
import torch.nn.functional as F
# Gives easier dataset management and creates mini batches
from torch.utils.data import DataLoader
# Has standard datasets we can import in a nice and easy way
import torchvision.datasets as datasets
# Transformations we can perform on our dataset
import torchvision.transforms as transforms


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 28x28 = 784 nodes
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape)
