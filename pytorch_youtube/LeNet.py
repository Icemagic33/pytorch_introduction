# Imports
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(
            5, 5), stride=(1, 1), padding=(1, 1))
        self.s2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=(5, 5), stride=())
