# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# VGG16 Architecture summary
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
         'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# then, flatten and 4096x4096x1000 linear layers


class VGG_16(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_16, self).__init__()
