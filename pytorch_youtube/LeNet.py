# Imports
import torch
import torch.nn as nn

# LeNet archintecture
# 1x32x32 Input -> (5x5),s=1,p=0 -> avg pool s=2,p=0 -> (5x5),s=1,p=0 -> avg pool s=2,p=0
# -> Conv 5x5 to 120 channels x Linear 84 x Linear 10


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(
            5, 5), stride=(1, 1), padding=(0, 0))
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120,
                            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)
