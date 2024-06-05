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
